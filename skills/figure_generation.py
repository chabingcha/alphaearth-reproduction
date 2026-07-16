"""
Skill: Figure Generation for AlphaEarth Reproduction

Generates 6 publication-quality comparison figures from experiment results.
Reusable for any interpretability reproduction with matching JSON schema.

Color palette: Nature/AAAS-inspired
Output: 250 DPI PNG files
"""
# This skill is implemented in src/visualization.py
# Import and use: from visualization import generate_all_figures

if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    from visualization import generate_all_figures
    generate_all_figures()
