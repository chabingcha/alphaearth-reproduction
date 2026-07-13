"""
FAISS vector index and Retrieval-Augmented Generation (RAG) pipeline.

Implements the Land Surface Intelligence (LSI) system described in the paper:
  - FAISS IndexIVFFlat for fast similarity search over embedding vectors
  - Location-based and environmental profile-based querying
  - RAG context assembly for LLM-powered environmental queries

Paper specs: IndexIVFFlat, nlist=3,500, 12.1M vectors, sub-millisecond latency
Our GBA specs: IndexIVFFlat, nlist=256, 234K vectors, ~100μs latency
"""

import numpy as np
import json
import time
from pathlib import Path

from .config import (
    FAISS_NLIST, FAISS_K_NEIGHBORS, FAISS_DIR, GBA_VAR_LIST, VAR_DISPLAY_NAMES,
)
from .preprocessing import normalize_embeddings


# ══════════════════════════════════════════════════════════════════
# FAISS Index Construction
# ══════════════════════════════════════════════════════════════════

def build_faiss_index(embeddings, index_type="IVFFlat", normalize=True,
                      nlist=None, metric="inner_product", verbose=True):
    """
    Build a FAISS index for fast similarity search over embedding vectors.

    Supports both raw (spatial context preserved) and debiased (environment-focused)
    index variants.

    Args:
        embeddings: (N, D) float32 array
        index_type: "IVFFlat" (paper) or "FlatL2" (exact)
        normalize: L2-normalize for cosine similarity
        nlist: Number of Voronoi cells (default: FAISS_NLIST)
        metric: "inner_product" (cosine on normalized vectors) or "l2"

    Returns:
        faiss_index: The built index
        metadata: Index metadata dict
    """
    try:
        import faiss
    except ImportError:
        print("  ⚠ FAISS not installed — install with: pip install faiss-cpu")
        return None, {"error": "faiss not installed"}

    N, D = embeddings.shape
    if nlist is None:
        nlist = min(FAISS_NLIST, int(np.sqrt(N)))  # Rule of thumb: ~√N

    # Normalize for cosine similarity
    if normalize:
        vectors = normalize_embeddings(embeddings.astype(np.float32), method="l2")
    else:
        vectors = embeddings.astype(np.float32).copy()

    if verbose:
        print(f"\n  Building FAISS Index:")
        print(f"    Vectors: {N:,} × {D}d")
        print(f"    Type: {index_type}, nlist={nlist}")
        print(f"    Metric: {metric}")

    t0 = time.time()

    # Quantizer (coarse clustering)
    quantizer = faiss.IndexFlatIP(D) if metric == "inner_product" else faiss.IndexFlatL2(D)

    if index_type == "IVFFlat":
        index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT
                                   if metric == "inner_product" else faiss.METRIC_L2)
    elif index_type == "FlatL2":
        index = faiss.IndexFlatL2(D)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    # Train (clustering for IVF)
    if index_type == "IVFFlat" and not index.is_trained:
        index.train(vectors)
        if verbose:
            print(f"    Training completed")

    # Add vectors
    index.add(vectors)
    build_time = time.time() - t0

    metadata = {
        "n_vectors": N,
        "dimension": D,
        "index_type": index_type,
        "nlist": nlist,
        "metric": metric,
        "normalized": normalize,
        "build_time_seconds": build_time,
    }

    if verbose:
        print(f"    Built in {build_time:.2f}s")
        print(f"    Total vectors indexed: {index.ntotal:,}")

    return index, metadata


# ══════════════════════════════════════════════════════════════════
# Query Engine
# ══════════════════════════════════════════════════════════════════

def query_similar_locations(index, query_vector, k=FAISS_K_NEIGHBORS):
    """
    Find k most similar locations to a query vector.

    Args:
        index: FAISS index
        query_vector: (D,) float32 or (1, D) array
        k: Number of neighbors

    Returns:
        distances: Cosine similarities (if inner_product) or L2 distances
        indices: Index positions of nearest neighbors
    """
    if index is None:
        return None, None

    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    query_vector = query_vector.astype(np.float32)

    # Normalize if using inner product
    norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    query_vector = query_vector / norms

    distances, indices = index.search(query_vector, k)
    return distances[0], indices[0]


def query_by_location(index, embeddings, coords, target_lat, target_lon,
                      k=FAISS_K_NEIGHBORS, env_vars=None, var_names=None):
    """
    Query FAISS index for locations similar to a given lat/lon.

    This is the primary LSI system interface — takes a location,
    finds similar locations in embedding space, and returns
    environmental context.

    Args:
        index: FAISS index
        embeddings: (N, D) all embeddings (for lookup)
        coords: (N, 2) [lat, lon]
        target_lat, target_lon: Query coordinates
        k: Number of similar locations
        env_vars: (N, M) environmental variables for context
        var_names: Variable names

    Returns:
        dict with query results and environmental context
    """
    # Find nearest grid point to target coordinates
    dist_to_target = np.sqrt((coords[:, 0] - target_lat)**2 +
                             (coords[:, 1] - target_lon)**2)
    nearest_idx = np.argmin(dist_to_target)
    query_vector = embeddings[nearest_idx]

    # FAISS search
    distances, indices = query_similar_locations(index, query_vector, k=k)

    if distances is None:
        return {"error": "FAISS index not available"}

    # Assemble results
    similar_locs = []
    for i, (d, idx) in enumerate(zip(distances, indices)):
        if idx < len(coords):
            loc_info = {
                "rank": i + 1,
                "similarity": float(d),
                "lat": float(coords[idx, 0]),
                "lon": float(coords[idx, 1]),
                "distance_km": float(
                    haversine_distance(target_lat, target_lon,
                                       coords[idx, 0], coords[idx, 1])
                ),
            }

            # Add environmental context if available
            if env_vars is not None and var_names is not None and idx < len(env_vars):
                loc_info["environment"] = {
                    var_names[j]: float(env_vars[idx, j])
                    for j in range(min(len(var_names), env_vars.shape[1]))
                }

            similar_locs.append(loc_info)

    return {
        "query": {"lat": target_lat, "lon": target_lon},
        "nearest_grid_point": {
            "lat": float(coords[nearest_idx, 0]),
            "lon": float(coords[nearest_idx, 1]),
            "distance_km": float(haversine_distance(
                target_lat, target_lon,
                coords[nearest_idx, 0], coords[nearest_idx, 1]
            )),
        },
        "similar_locations": similar_locs,
        "mean_similarity": float(np.mean(distances)),
    }


def query_by_profile(index, embeddings, env_vars, coords, profile_spec,
                     var_names=None, k=FAISS_K_NEIGHBORS):
    """
    Find locations matching an environmental profile.

    Instead of querying by location, query by desired environmental
    characteristics (e.g., "cool and wet").

    This uses the debiased FAISS index (environment-focused) for
    cross-location analog finding.

    Args:
        index: FAISS index (preferably debiased for env focus)
        embeddings: (N, D)
        env_vars: (N, M)
        coords: (N, 2)
        profile_spec: dict mapping var_name → desired value
                      e.g., {"t_air_mean": 290, "precip": 0.005}
        var_names: All variable names
        k: Number of matches

    Returns:
        dict with matching locations and their environmental profiles
    """
    if var_names is None:
        var_names = GBA_VAR_LIST

    # Build a synthetic embedding from the profile
    # Use a weighted combination of embeddings at locations with matching profiles
    M = env_vars.shape[1]
    scores = np.ones(len(embeddings))

    for var, target_val in profile_spec.items():
        if var in var_names:
            j = var_names.index(var)
            # Gaussian similarity based on how close env var is to target
            var_std = np.std(env_vars[:, j]) + 1e-8
            similarity = np.exp(-0.5 * ((env_vars[:, j] - target_val) / var_std) ** 2)
            scores *= similarity

    # Use top-scoring location's embedding as query
    best_idx = np.argmax(scores)
    query_vector = embeddings[best_idx]

    distances, indices = query_similar_locations(index, query_vector, k=k)

    if distances is None:
        return {"error": "FAISS index not available"}

    matches = []
    for i, (d, idx) in enumerate(zip(distances, indices)):
        if idx < len(coords):
            match_info = {
                "rank": i + 1,
                "similarity": float(d),
                "lat": float(coords[idx, 0]),
                "lon": float(coords[idx, 1]),
            }
            if env_vars is not None and idx < len(env_vars):
                match_info["environment"] = {
                    var_names[j]: float(env_vars[idx, j])
                    for j in range(min(len(var_names), env_vars.shape[1]))
                }
            matches.append(match_info)

    return {
        "query_profile": profile_spec,
        "best_match_idx": int(best_idx),
        "matches": matches,
    }


# ══════════════════════════════════════════════════════════════════
# RAG Context Assembly
# ══════════════════════════════════════════════════════════════════

def build_rag_context(query_results, dimension_dictionary=None,
                      var_names=None, top_k=10):
    """
    Assemble a RAG context string from FAISS search results and
    dimension dictionary for use in LLM prompting.

    This is the 5-stage pipeline from the paper:
      1. Location Resolution
      2. Embedding Retrieval
      3. Dimension Interpretation (Dimension Dictionary)
      4. Context Assembly (FAISS k-NN + dimension mappings)
      5. LLM Generation

    Args:
        query_results: Output from query_by_location or query_by_profile
        dimension_dictionary: Dim→env_var mapping
        var_names: Variable names for display
        top_k: How many similar locations to include

    Returns:
        str: Formatted RAG context for LLM prompt
    """
    if var_names is None:
        var_names = GBA_VAR_LIST

    parts = []

    # Location context
    if "query" in query_results:
        q = query_results["query"]
        parts.append(f"## Query Location\n")
        parts.append(f"- Latitude: {q['lat']:.4f}°N\n")
        parts.append(f"- Longitude: {q['lon']:.4f}°E\n\n")

    # Similar locations
    if "similar_locations" in query_results:
        parts.append(f"## {min(top_k, len(query_results['similar_locations']))} Most Similar Locations\n\n")
        parts.append("| Rank | Similarity | Lat | Lon | Distance (km) | Key Env Vars |\n")
        parts.append("|------|-----------|-----|-----|---------------|-------------|\n")

        for loc in query_results["similar_locations"][:top_k]:
            env_str = ""
            if "environment" in loc:
                top_vars = sorted(loc["environment"].items(),
                                  key=lambda x: abs(x[1]), reverse=True)[:3]
                env_str = ", ".join(f"{k}={v:.3f}" for k, v in top_vars)

            parts.append(
                f"| {loc['rank']} | {loc['similarity']:.4f} | "
                f"{loc['lat']:.4f} | {loc['lon']:.4f} | "
                f"{loc.get('distance_km', 0):.1f} | {env_str} |\n"
            )
        parts.append("\n")

    # Dimension dictionary context
    if dimension_dictionary:
        parts.append("## Dimension-Environment Mappings\n\n")
        parts.append("Key embedding dimensions and their associated environmental variables:\n\n")
        parts.append("| Dimension | Primary Variable | Spearman ρ | Concordance |\n")
        parts.append("|-----------|-----------------|------------|-------------|\n")

        # Show top dimensions by Spearman |ρ|
        sorted_dims = sorted(
            dimension_dictionary.items(),
            key=lambda x: abs(x[1].get("spearman_rho", 0)),
            reverse=True,
        )[:10]

        for dim_name, info in sorted_dims:
            rho = info.get("spearman_rho", 0)
            concordant = "✓" if info.get("concordant") else ""
            parts.append(
                f"| {dim_name} | {info.get('primary_var', 'unknown')} | "
                f"{rho:+.3f} | {concordant} |\n"
            )
        parts.append("\n")

    # Assembly instructions for LLM
    parts.append("## Task\n\n")
    parts.append(
        "Based on the above information, provide a grounded environmental assessment "
        "of the query location. Include:\n"
        "1. Summary of key environmental characteristics\n"
        "2. How it compares to similar locations\n"
        "3. Any notable patterns in the dimension-variable mappings\n"
        "Cite specific values from the context above.\n"
    )

    return "".join(parts)


# ══════════════════════════════════════════════════════════════════
# Demo Queries
# ══════════════════════════════════════════════════════════════════

def run_demo_queries(index, embeddings, env_vars, coords,
                     var_names=None, verbose=True):
    """
    Run demonstration queries on the LSI system.

    Three query types matching the report:
      1. Location query (Guangzhou)
      2. Environmental profile query (cool & wet)
      3. Cross-location analog (Shenzhen)

    Returns:
        dict with all demo query results
    """
    if index is None:
        return {"error": "FAISS index not available for demo"}

    if var_names is None:
        var_names = GBA_VAR_LIST

    demos = {}

    # Demo 1: Location query — Guangzhou
    if verbose:
        print("\n  Demo 1: Location Query (Guangzhou, 23.13°N, 113.26°E)")
    guangzhou = query_by_location(
        index, embeddings, coords,
        target_lat=23.13, target_lon=113.26,
        k=5, env_vars=env_vars, var_names=var_names,
    )
    demos["guangzhou"] = guangzhou

    if verbose and "similar_locations" in guangzhou:
        sims = [loc["similarity"] for loc in guangzhou["similar_locations"]]
        print(f"    Retrieved {len(sims)} locations, similarities: {[f'{s:.3f}' for s in sims]}")
        if "environment" in guangzhou["similar_locations"][0]:
            env = guangzhou["similar_locations"][0]["environment"]
            t_air = env.get("t_air_mean", "N/A")
            print(f"    Top match: similarity={sims[0]:.4f}, t_air_mean={t_air}")

    # Demo 2: Environmental profile query — cool and wet
    if verbose:
        print("\n  Demo 2: Environmental Profile Query (cool & wet)")
    cool_wet = query_by_profile(
        index, embeddings, env_vars, coords,
        profile_spec={"t_air_mean": 290.0, "precip": 0.005},
        var_names=var_names, k=5,
    )
    demos["cool_wet"] = cool_wet

    if verbose and "matches" in cool_wet:
        print(f"    Best match: ({cool_wet.get('best_match_idx', 'N/A')})")
        if cool_wet["matches"]:
            m = cool_wet["matches"][0]
            t_air = m.get("environment", {}).get("t_air_mean", "N/A")
            precip = m.get("environment", {}).get("precip", "N/A")
            print(f"    t_air_mean={t_air}, precip={precip}")

    # Demo 3: Cross-location analog — Shenzhen
    if verbose:
        print("\n  Demo 3: Cross-Location Analog (Shenzhen, 22.54°N, 114.06°E)")
    shenzhen = query_by_location(
        index, embeddings, coords,
        target_lat=22.54, target_lon=114.06,
        k=10, env_vars=env_vars, var_names=var_names,
    )
    demos["shenzhen"] = shenzhen

    if verbose and "similar_locations" in shenzhen:
        sims = [loc["similarity"] for loc in shenzhen["similar_locations"]]
        # Count how many similar locations are geographically close to Shenzhen
        nearby = sum(1 for loc in shenzhen["similar_locations"]
                    if abs(loc["lat"] - 22.54) < 0.5 and abs(loc["lon"] - 114.06) < 0.5)
        print(f"    Retrieved {len(sims)} locations, {nearby} within 0.5° of Shenzhen")
        print(f"    Similarity range: {min(sims):.3f}–{max(sims):.3f}")

    return demos


# ══════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════

def haversine_distance(lat1, lon1, lat2, lon2):
    """Compute distance in km between two lat/lon points using Haversine formula."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c


def save_faiss_index(index, name="gba_index"):
    """Save FAISS index to disk."""
    try:
        import faiss
    except ImportError:
        return

    path = FAISS_DIR / f"{name}.faiss"
    faiss.write_index(index, str(path))
    print(f"  [OK] FAISS index saved: {path}")


def load_faiss_index(name="gba_index"):
    """Load FAISS index from disk."""
    try:
        import faiss
    except ImportError:
        return None

    path = FAISS_DIR / f"{name}.faiss"
    if path.exists():
        return faiss.read_index(str(path))
    return None
