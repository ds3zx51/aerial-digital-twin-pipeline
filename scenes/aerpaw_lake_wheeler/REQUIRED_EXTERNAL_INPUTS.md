# Required external artifacts

Formal `paper_reconstruction` requires artifacts that are not licensed or present here:

1. A verified AODT scene bundle containing the Lake Wheeler terrain, buildings, and foliage.
2. The paper's supplementary foliage GeoJSON, or an evidence-equivalent digitization with its
   own provenance record.
3. A georeferencing manifest defining CRS, origin, axis order, units, vertical datum, scene
   transform, and terrain heights for all UAV/tower samples.

Do not rename the empty `supplementary_foliage.template.geojson` as evidence. BU04 and BU07
detect empty/template layers and reject them for formal use.
