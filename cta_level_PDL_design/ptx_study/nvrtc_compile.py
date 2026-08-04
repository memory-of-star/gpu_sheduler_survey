#!/usr/bin/env python3
# Drive NVRTC via ctypes to compile a .cu to PTX and print it.
import ctypes, sys, glob, os

def find_libnvrtc():
    pattern = "~/.local/lib/python3*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*"
    hits = glob.glob(os.path.expanduser(pattern))
    hits = [h for h in hits if not h.endswith("alt.so") and "builtins" not in h]
    if not hits:
        sys.exit("libnvrtc not found")
    return sorted(hits)[-1]

def main():
    src_path = sys.argv[1] if len(sys.argv) > 1 else "pdl_demo.cu"
    arch = sys.argv[2] if len(sys.argv) > 2 else "compute_90"
    with open(src_path, "rb") as f:
        src = f.read()

    lib = ctypes.CDLL(find_libnvrtc())

    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(ctypes.byref(prog), src, src_path.encode(), 0, None, None)
    if rc != 0:
        sys.exit(f"nvrtcCreateProgram failed rc={rc}")

    opts = [f"--gpu-architecture={arch}".encode(), b"-default-device"]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)

    # Always fetch the log.
    log_size = ctypes.c_size_t()
    lib.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    lib.nvrtcGetProgramLog(prog, log)
    sys.stderr.write("=== NVRTC LOG ===\n" + log.value.decode(errors="replace") + "\n")

    if rc != 0:
        sys.exit(f"nvrtcCompileProgram failed rc={rc}")

    ptx_size = ctypes.c_size_t()
    lib.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    lib.nvrtcGetPTX(prog, ptx)
    sys.stdout.write(ptx.value.decode(errors="replace"))

if __name__ == "__main__":
    main()
