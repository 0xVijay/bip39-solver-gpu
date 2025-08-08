// List of supported GPU models for CUDA and OpenCL backends
pub const SUPPORTED_GPU_MODELS: &[&str] = &[
    // NVIDIA Ada, Hopper, Ampere, Turing, Volta, Pascal, Maxwell, Kepler, Fermi, etc.
    "A800 PCIe", "A100 PCIe", "A100 SXM4", "A100 SXM", "A100X", "B200", "GH200 SXM", "H100 NVL", "H100 PCIe", "H100 SXM", "H200 NVL", "H200 SXM",
    "A40", "A30", "A16", "A10G", "A10", "L40S", "MIbI L40", "L4", "Tesla K80", "Tesla P100", "Tesla P40", "Tesla P4", "Tesla T4", "Tesla V100",
    "GTX 1660 Ti", "GTX 1660 Super", "GTX 1660", "GTX 1650 Super", "GTX 1650", "GTX 1080 Ti", "GTX 1080", "GTX 1070 Ti", "GTX 1070", "GTX 1060", "GTX 1050 Ti", "GTX 1050", "GTX 980 Ti", "GTX 980", "GTX 970", "GTX 960", "GTX 750 Ti", "GTX 750",
    "RTX 8000", "RTX 6000", "RTX 5000", "RTX 4000", "GP100", "P6000", "P5000", "P4000", "P2000", "P106-100", "P104-100", "Titan RTX", "Titan V", "Titan X", "Titan Xp",
    "RTX 5090", "RTX 5080", "RTX 5070 Ti", "RTX 5070", "RTX 5060 Ti", "RTX 4090 D", "RTX 4090", "RTX 4080 Super", "RTX 4080", "RTX 4070 Ti Super", "RTX 4070 Ti", "RTX 4070 Super", "RTX 4070", "RTX 4060 Ti", "RTX 4060", "RTX 4060 Laptop", "RTX 3090 Ti", "RTX 3090", "RTX 3080 Ti", "RTX 3080", "RTX 3070 Ti", "RTX 3070", "RTX 3070 Laptop", "RTX 3060 Ti", "RTX 3060", "RTX 3060 Laptop", "RTX 3050",
    "RTX 2080 Ti", "RTX 2080 Super", "RTX 2080", "RTX 2070 Super", "RTX 2070", "RTX 2060 Super", "RTX 2060",
    "RTX PRO 6000 Blackwell Workstation", "RTX 6000 Ada Generation", "RTX 5880 Ada Generation", "RTX 5000 Ada Generation", "RTX 4500 Ada Generation", "RTX 4000 Ada Generation", "RTX A6000", "RTX A5000", "RTX A4500", "RTX A4000", "RTX A2000",
    // AMD, Intel, and others can be added here
];
