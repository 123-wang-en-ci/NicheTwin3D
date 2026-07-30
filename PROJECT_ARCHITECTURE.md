# ShowSToFM2.0 项目架构文档

## 项目概述
ShowSToFM2.0 是一个基于Unity的空间转录组学数据可视化系统，结合了Unity前端可视化与Python后端AI模型处理。该项目旨在提供一个交互式的平台，让用户能够探索和分析空间转录组数据。

## 整体架构
```
┌─────────────────────────────────────────────────────────────────┐
│                    Unity Frontend Client                        │
├─────────────────────────────────────────────────────────────────┤
│  UI_GeneSearch.cs   UIManager.cs    SimpleCameraController.cs  │
│  InteractionManager.cs  TooltipController.cs                   │
│  DataLoaderGPU.cs    GPURenderer.cs   CellProxyManager.cs      │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ HTTP Requests
┌─────────────────────────────────────────────────────────────────┐
│                    Python Backend Server                        │
│                     (FastAPI Framework)                         │
├─────────────────────────────────────────────────────────────────┤
│  server.py          model_engine.py   Nicheformer Models       │
│  analyze_tissue_segmentation.py  diagnose_clustering.py         │
│  evaluate_model.py  train_downstream.py  train_nicheformer.py   │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ AI Processing
┌─────────────────────────────────────────────────────────────────┐
│                    Nicheformer AI Engine                        │
│              (Spatial Transcriptomics Analysis)                 │
├─────────────────────────────────────────────────────────────────┤
│  - Spatial Graph Construction     - Gene Expression Imputation │
│  - Cell Type Classification      - Spatial Clustering          │
│  - Perturbation Simulation       - Embedding Generation        │
└─────────────────────────────────────────────────────────────────┘
```

## 组件详细说明

### Unity 前端模块

#### 1. UI 模块
- **UI_GeneSearch.cs**: 基因搜索界面，允许用户输入基因名称并请求相应数据
- **UIManager.cs**: 管理UI元素的状态和交互
- **TooltipController.cs**: 显示细胞详情的提示框

#### 2. 渲染模块
- **GPURenderer.cs**: 使用GPU实例化技术渲染大量细胞，避免传统GameObject的性能瓶颈
- **DataLoaderGPU.cs**: 负责从后端加载数据并将其传递给GPU渲染器
- **CellProxyManager.cs**: 为GPU渲染的细胞创建碰撞体，支持物理交互检测

#### 3. 交互控制模块
- **InteractionManager.cs**: 处理用户与数据的交互，包括基因切换、细胞扰动等
- **CameraOrbit.cs / SimpleCameraController.cs**: 控制相机视角，支持缩放、旋转和平移

### Python 后端模块

#### 1. 服务框架
- **server.py**: FastAPI应用，提供REST API接口
- **CORS中间件**: 支持跨域请求，使Unity客户端可访问后端服务

#### 2. AI 引擎
- **model_engine.py**: Nicheformer引擎，封装AI模型推理功能
- **Nicheformer模型**: 基于Transformer的深度学习模型，用于空间转录组数据分析

#### 3. 数据处理
- **H5AD文件**: 存储空间转录组数据的标准格式
- **Scanpy**: 用于读取和处理单细胞数据的Python库

## 数据流

### 1. 初始化流程
```
1. Unity启动 → DataLoaderGPU加载CSV数据
2. GPURenderer渲染初始细胞布局
3. CellProxyManager创建碰撞体代理
4. Python后端启动 → 加载H5AD数据 → 初始化Nicheformer模型
```

### 2. 基因搜索流程
```
1. 用户在UI输入基因名称
2. UI_GeneSearch.cs发送请求到InteractionManager.cs
3. InteractionManager.cs向Python后端发起API请求
4. server.py接收请求，通过DataManager查询Nicheformer引擎
5. Nicheformer进行基因表达预测和插补
6. 返回JSON格式的结果到Unity
7. DataLoaderGPU.cs解析结果并更新GPU缓冲区
8. GPURenderer.cs重新渲染细胞（颜色/高度变化）
```

### 3. 交互流程
```
1. 用户点击细胞
2. CellProxyManager.cs检测到碰撞
3. InteractionManager.cs获取细胞详情
4. TooltipController.cs显示细胞信息
5. 可选：发送扰动请求到后端
6. 后端处理扰动模拟
7. 结果返回并更新可视化
```

## 技术栈

### 前端 (Unity)
- **C#**: 主要编程语言
- **Unity 3D**: 游戏引擎用于数据可视化
- **GPU Instancing**: 高性能渲染技术
- **Shader**: 自定义着色器用于渲染效果

### 后端 (Python)
- **FastAPI**: 现代高性能Web框架
- **PyTorch**: 深度学习框架
- **Scanpy**: 单细胞数据处理
- **Numpy/Pandas**: 数据处理
- **Scikit-learn**: 机器学习算法

### AI模型
- **Nicheformer**: 专门为空间转录组设计的深度学习模型
- **Transformer架构**: 用于捕获空间关系
- **空间图神经网络**: 分析细胞间的空间相互作用

## 文件结构
```
Server/
├── Nicheformer/          # Nicheformer模型源码
├── data/                 # 数据文件
├── server.py             # 主服务文件
├── model_engine.py       # AI模型引擎
├── *.h5ad                # 空间转录组数据
└── *.pth                 # 训练好的模型权重
Assets/Scripts/
├── GPURenderer.cs        # GPU实例化渲染
├── DataLoaderGPU.cs      # 数据加载
├── InteractionManager.cs # 交互管理
├── UI_GeneSearch.cs      # 基因搜索UI
└── *.md                  # 项目文档
```

## 关键特性

1. **高性能渲染**: 使用GPU实例化技术处理大量细胞数据
2. **AI增强分析**: Nicheformer模型提供基因表达插补和预测
3. **实时交互**: 支持基因搜索、细胞选择、扰动模拟等功能
4. **空间感知**: 保留和展示细胞的空间位置关系
5. **可扩展性**: 模块化设计支持添加新功能

## 开发环境要求

### 前端
- Unity 2021.3 LTS 或更高版本
- C# 8.0+ 编译器

### 后端
- Python 3.8+
- PyTorch 1.12+
- FastAPI 0.68+

### 依赖库
- Scanpy
- Pandas
- NumPy
- Scikit-learn
- Transformers