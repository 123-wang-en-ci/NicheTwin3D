# Changelog

## [1.6.1]

### Fixed

- **[Backend] Model loading "Silent Failure Fail-Fast"**: Deeply fixed the serious flaw that when loading pre-training weights under a custom data set (non-default 32,101 genes), an exception is thrown by PyTorch due to a mismatch in the Embedding layer dimension, but the system forcibly continues and outputs the disguise success. An explicit `sys.exit(1)` interception has been added to `model_engine.py`. Once a Size Mismatch is detected, the service will be immediately interrupted and an intuitive error will be thrown to the console.
- **[Backend] Pre-training vocabulary destructive coverage repair**: Cancel the logic of violently overwriting the system global `gene_vocab.npy` when a discrepancy in the number of custom H5AD genes is detected. Now the system will only apply the gene list of the current data set in memory, completely protecting the original pre-trained model weight vocabulary from being contaminated.

## [1.6.0]

### Added

- **[Frontend] Modular left navigation and right dynamic placement slot architecture: **Newly reconstructed UI visual layout, abstracting the six core main navigation buttons on the left (gene expression interpolation, cell type annotation, regional organization semantic segmentation, zero-sample clustering, settings, help). Combined with the new `TabNavigationManager` controller and the dynamic content slot on the right (`RightContentSlot`), the secondary sub-function group on the right can be seamlessly switched and displayed when the left button is clicked.
- **[Frontend] resident public toolbar component decoupling:** Detach system-level public keys like '[Screenshot (F12)]' and '[Reset View]' into independent resident slots ('panel_SharedCommon'), completely avoiding UI redundancy that requires repeated creation and binding across multiple subpanels; It also intelligently hides when switching to the "Settings" or "Help" panels, keeping the interface clean.
- **[Frontend] Freely dragging and border-scaling the help window: **Added window border drag-and-retract and TitleBar drag controllers to the help document pop-up window, allowing users to freely adjust the position, width and height of the help window just like operating a standard operating system window, and the internal text maintains adaptive responsive typesetting.

### Changed

- **[Frontend] Help documentation refinement and precise button matching:** Overload the Chinese and English 'HelpManager' help guide, precisely mapping each operation step to the corresponding interface button and input box name (e.g., '[Enter gene...]' ', '[AI gene expression interpolation]', '[Curved Mode: Off]', '[Contrast Mode: On]'). At the same time, the actual operation order for cell type annotation and region segmentation has been corrected: "first click the button to run AI prediction, then pull down to select classification."
- **[Frontend] Language Toggle Button Copywriting Standardization:** Unified bilingual switch button text across the entire interface is '[Chinese / English]' (CN) and '[EN / CN]' (EN), enhancing the visual appeal of prompts.

### Fixed

- **[Frontend] UI panel mouse hover ray penetration isolation:** Added a global 'EventSystem.IsPointerOverGameObject()' hover detection defense line to 'TooltipController', completely solving the visual conflict issue where the background 3D cell ID prompt box passes through the mouse when sliding over the left navigation bar or the right content slot.
- **[Frontend] 3D Camera Launch Smoothing and WASD Jump Fix:** Deeply fixed issues where camera zooms in abnormally when the software starts and when pressing or releasing the WASD movement key causes the image to instantly jump upward (Jump Up). In 'CameraOrbit', the target focus initialization and full real-time Euler angle (yaw/pitch) synchronization mechanism have been reconstructed, achieving seamless and smooth alignment of coordinate systems between WASD panning, scroll wheel scaling, and right-click rotation.

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