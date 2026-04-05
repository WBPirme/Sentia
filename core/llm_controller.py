import os
import subprocess
import socket
import platform
import asyncio
import aiohttp  # ⚠️ 终极解法：使用纯异步的 HTTP 库代替阻塞的 urllib！


class LlamaEngineController:
    def __init__(self, base_dir, port=8099, model_name="Sentia-Q4_K_M.gguf"):
        self.base_dir = base_dir
        self.engine_root = os.path.join(base_dir, "engine")
        self.model_path = os.path.join(base_dir, "models", model_name)
        self.port = port
        self.process = None

    def _sniff_hardware(self):
        if platform.system() != "Windows": return "llama.cpp-cpu"
        try:
            output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode().upper()
            if "NVIDIA" in output: return "llama.cpp-cuda12"
            if "AMD" in output or "RADEON" in output: return "llama.cpp-hip"
            if "INTEL" in output: return "llama.cpp-vulkan"
        except:
            pass
        return "llama.cpp-cpu"

    def is_port_in_use(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', self.port)) == 0

    async def wait_for_server_ready(self, timeout=60):
        print(f"⏳ 正在等待大模型载入显存 (端口 {self.port})...")
        start_time = asyncio.get_event_loop().time()
        url = f"http://localhost:{self.port}/health"

        async with aiohttp.ClientSession() as session:
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    async with session.get(url, timeout=1) as response:
                        if response.status == 200:
                            print(
                                f"\n 大模型已就绪！(耗时: {asyncio.get_event_loop().time() - start_time:.1f}秒)\n")
                            return True
                except:
                    await asyncio.sleep(0.5)

        print("\n 警告：大模型启动超时！")
        return False

    def start_process_only(self):
        if self.is_port_in_use():
            print(f" 大模型已在运行。")
            return

        engine_folder = self._sniff_hardware()
        target_engine_dir = os.path.join(self.engine_root, engine_folder)
        exe_path = os.path.join(target_engine_dir, "llama-server.exe")

        print(f" 正在加载 {engine_folder} ...")
        layers = "99" if engine_folder != "llama.cpp-cpu" else "0"

        cmd = [
            exe_path, "-m", self.model_path, "-ngl", layers,
            "--port", str(self.port), "--chat-template", "chatml"
        ]

        env = os.environ.copy()
        env["PATH"] = target_engine_dir + os.pathsep + env.get("PATH", "")

        self.process = subprocess.Popen(
            cmd, cwd=target_engine_dir, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

    def stop(self):
        if self.process:
            self.process.terminate()
            print("  llama.cpp 已退出。")