"""Stone Completion Pipeline -- PointSea-inspired completion + NKSR mesh.

End-to-end pipeline for stone volume estimation:
  1. Segment stone from floor (PointNet++ + per-point classifier)
  2. Complete partial stone surface (PointSea-style self-view fusion + SDG)
  3. Reconstruct watertight mesh (NKSR pre-trained, inference only)
  4. Compute volume geometrically from mesh
"""
