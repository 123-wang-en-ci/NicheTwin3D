# Changelog

## [1.5.0]

## Added

- **[Frontend] Dual-screen real-time comparison system: **Newly created dual-screen comparison mode, the left side renders the original base state, and the right side renders the predicted result. With the intelligent left screen anti-accidental touch lock and the non-delay physical synchronization alignment of the secondary camera, intuitive sequencing and predictive micro-crossing feedback are provided.
- **[Frontend] Gene Intelligent Fuzzy Search:** Adds fuzzy association search functionality to gene search interaction. The system intelligently recommends a list of the top 50 candidate genes based on the missing keywords entered by the user, entering them with one click, completely avoiding manual spelling errors of complex gene names (such as Ensembl ID).
- **[Frontend] Clustering Parameter Quick Dropdown:** For parameter input in zero-shot clustering, a new smart preset panel has been added. Clicking the input box immediately brings up 5 biologically verified classic resolution shortcuts, perfectly preserving the ability to customize plain text input.
- **[Frontend] F12 shortcut to export pictures with one click: **The `F12` shortcut key is globally bound to the high-quality "Export Picture" function. It allows users to right-click and freely roam the 3D space to find the best camera position. At the same time, they can hide all environmental UIs with one click and complete the capture instantly.

## Changed

- **[Frontend] Rigorous algorithm loading text: **The front-end loading prompt text when running zero-sample clustering is completely revised from the historical `Running K-Means Clustering` to the advanced algorithm `Running Leiden Clustering` that is actually run at the bottom to ensure the consistency of academic expression.
- **[Frontend] Retrieval Interaction Mistouch Prevention Refactor:** Optimized the trigger logic for gene fuzzy search dropdown menus. Clicking a dropdown menu item only performs the "Fill Input Box and Collapse Panel" operation; you must manually click the search button to trigger a network request, greatly improving the sense of control and fault tolerance of the operation.

## Fixed

- **[Frontend] Camera perspective "bounce" fix: **Depth has fixed the vicious flaw that the screen will instantly jump back to the old coordinates when using the right mouse button to rotate / adjust the perspective. By forcing synchronization of underlying Euler angles and depth distances at the right-click frame, a completely coherent and silky-smooth 3D roaming experience has been recreated.
- **[Frontend] Input method and space roaming conflict fix: **Globally introduced EventSystem ray anti-penetration detection. Completely fixed the problem that typing in UI input box (e.g. typing w,s,a,d) would cause the camera to fly around in the background, and the conflict that right-clicking on UI panel triggered the viewport to rotate incorrectly.

## [1.4.0]

### Added

- **[Backend]** **Console gene search guidance**: When the backend starts loading the model, it will automatically print out the list of valid genes contained in the data set to the console. Users can directly copy the names to the frontend for accurate search.
- **[Backend]** **Console gene search guidance**: When the backend starts loading the model, it will automatically print out the list of valid genes contained in the data set to the console. Users can directly copy the names to the frontend for accurate search.

### Changed

- **[Frontend]** **Window Mode Refactoring**: Changes the software's basic display mode from forced fullscreen (exclusive mode) to **flexible window mode**. It completely solves the pain point where users cannot perform other research multitasking after opening the software and can only force exit via ESC.
- **[Backend]** Comprehensive internationalization of code and logs: In preparation for open source release and cross-border collaboration, the core comments, warning messages and console print logs of the backend have all been rewritten from Chinese translations to standard English.
- **[Backend]** **Clustering API Academic Standardization**: The request parameters for the zero-shot clustering interface are reconstructed from 'n_clusters' to 'resolution', directly exposing the underlying Leiden algorithm parameters and fully adapting to academic research standards in the single-cell sequencing field.

### Fixed

- **[Model & UI]** **Cellular Space Mapping and Classification Repair**: Fixed issues with inaccurate clicking and misaligned information in 3D space (e.g., incorrectly displaying 'pericytes' as 'pericells' when clicked). By improving the accuracy of the underlying software model, it now fully aligns and meets the standard accuracy of the original paper model.

## [1.3.0]

### Added

- **[Backend]** **Dual-track Data Stream Design**: Interface communication now supports simultaneously returning two sets of data—the "relative representation" required for rendering (controlling 3D bar height) and the real "sequencing counts value" (used for UI information panel display), balancing visual presentation with the rigor of biological data.
- **[Backend]** **Percentile Stretch Alignment Strategy**: Percentile Scaling is introduced in the gene interpolation module to ensure that the interpolation results produced by the Nicheformer latent space are perfectly consistent in magnitude with the real sequencing data, preventing 3D rendering from being highly out of control.
- **[Frontend]** **Interpolation Highlighting Visual Feedback**: Added independent color layer logic. Virtual gene data generated after gene expression interpolation now uses special "yellow" highlighting, creating a sharp contrast with the original sequencing data and solving the problem of interpolation results being difficult to distinguish with the naked eye.

## [1.2.0]

### Added

- **[Backend]** **SToFM Core Architecture Integration**: Fully introduces the Spatial Transcriptomics Foundation Model architecture, significantly raising the ceiling for spatial feature processing.
- **[Backend]** **Adaptive graph fusion interpolation**: The Adaptive Graph Imputation algorithm is introduced, using the cosine similarity of cells in the embedding space for feature diffusion, completely replacing the original backward "spatial geometric mean smoothing".

### Changed

- **[Backend]** **Classifier Residual Upgrade**: The `ClassifierHead` of downstream feature extraction is upgraded to a residual network (Skip Connection) structure with LayerNorm and GELU.
- **[Backend]** **Zero-Shot Clustering Algorithm Replacement**: Completely abandoning KMeans, fully adopting the industry's highest standard **KNN + Leiden community discovery algorithm**.

## [1.1.0]

### Added

- **[System]** **Standardized environment deployment**: Add the `environment.yml` file to achieve unified configuration of the Conda environment and avoid local dependency conflicts.