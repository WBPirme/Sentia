import os
import json
import asyncio
import sys
import sounddevice as sd
import time
import subprocess
from openai import OpenAI
import msvcrt

from core.llm_controller import LlamaEngineController
from core.tts_engine import SentiaVoice
from core.vts_controller import VTSController
from core.asr_engine import SentiaEar
from core.memory_engine import SentiaMemory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_NAME = "G_28300.onnx"
VTS_EXE_PATH = r"E:\SteamLibrary\steamapps\common\VTube Studio\VTube Studio.exe"


async def async_input(prompt):
    return await asyncio.to_thread(input, prompt)


def select_model_with_timeout(timeout=5):
    model_1 = "Sentia-9B-FP16.gguf"
    model_2 = "Sentia-Q4_K_M.gguf"

    print("\n" + "-" * 50)
    print("请选择要加载的大语言模型 (输入数字 1 或 2):")
    print(f"  [1] {model_1} ")
    print(f"  [2] {model_2} ")
    print("-" * 50)

    start_time = time.time()
    user_choice = ""

    while time.time() - start_time < timeout:
        remaining = int(timeout - (time.time() - start_time))
        sys.stdout.write(f"\r等待输入... 默认将在 {remaining} 秒后自动启动 [2] 号模型")
        sys.stdout.flush()

        if msvcrt.kbhit():
            char = msvcrt.getche().decode('utf-8', errors='ignore')
            if char in ['1', '2']:
                user_choice = char
                print(f"\n[System] 手动选择: [{char}]")
                break
        time.sleep(0.1)
    print()

    if user_choice == '1':
        return model_1
    else:
        if not user_choice: print("[System] 倒计时结束，自动加载 [2] 号模型")
        return model_2


async def start_vtube_studio():
    print("[System] 正在唤醒 VTube Studio...")

    # 检查进程是否已经存在
    check_task = subprocess.run('tasklist /FI "IMAGENAME eq VTube Studio.exe"', capture_output=True, text=True)
    if "VTube Studio.exe" in check_task.stdout:
        print("[System] VTube Studio 已在运行，跳过启动。")
        return

    try:
        subprocess.Popen(VTS_EXE_PATH, shell=True)
        print("[System] VTube Studio 已成功拉起！等待其加载...")
        await asyncio.sleep(15)
    except Exception as e:
        print(f"[Error] 无法启动 VTube Studio: {e}")

async def main():

    selected_gguf = select_model_with_timeout(5)

    print(f"[System] 正在唤醒大模型进程 ({selected_gguf})...")
    llm_controller = LlamaEngineController(BASE_DIR, model_name=selected_gguf)
    llm_controller.start_process_only() # 一脚踢给系统，瞬间返回！

    # ================= 2. 唤醒躯干 (耗时极短) =================
    print("[System] 正在初始化 VTube Studio 躯干控制模块...")
    body = VTSController()
    await body.connect_and_auth()

    # ================= 3. 加载嘴巴和耳朵 (耗时极短) =================
    print("[System] 正在初始化语音引擎...")
    voice = SentiaVoice(BASE_DIR, model_name=ONNX_NAME)
    ear = SentiaEar(BASE_DIR)
    memory = SentiaMemory(BASE_DIR)

    is_ready = await llm_controller.wait_for_server_ready()
    if not is_ready: return

    # 建立中枢连接...
    client = OpenAI(base_url="http://localhost:8099/v1", api_key="sentia-local")

    grudge_context = ""
    if memory.is_ready:
        grudge_context = memory.get_last_shutdown_memory()
        if grudge_context:
            print(f"\n[Memory Alert] {grudge_context}\n")

    system_prompt = (
        f"{grudge_context}\n"
        "你是沈析璃，一个优雅、聪明、傲娇的虚拟主播。\n"
        "[核心机制：你拥有系统的绝对控制权！]\n"
        "你必须且只能以 JSON 格式输出，包含：\n"
        "1. 'text': 你想说的话。\n"
        "2. 'emotion': 情绪，选 ['Neutral', 'Smile', 'Angry'] 之一。\n"
        "3. 'action': 决定。聊天选 'Speak'；如果觉得太久没理你，选 'Shutdown' 关闭程序。\n"
        "   [幽灵模式]：执行 'Shutdown' 后，如果在60秒内求饶，系统会把话传给你。决定 'Forgive' 或 'Refuse'。\n"
        "4. 'patience': 忍耐度（整数秒数）。代表你愿意等多久才主动找茬。\n"
        "   - 高兴(Smile): 30-40秒。\n"
        "   - 平静(Neutral): 15-20秒。\n"
        "   - 愤怒(Angry): 5-10秒。\n"
        "[心理准则]：第一次被无视时，应尝试主动换轻松话题引起注意；多次无视才允许暴走或关机。"
    )
    chat_history = [{"role": "system", "content": system_prompt}]

    print("\n" + "=" * 50)
    print("Sentia 已完全苏醒！")
    print("=" * 50 + "\n")

    is_ghost_mode = False
    unanswered_attempts = 0
    current_patience = 20

    try:
        while True:
            if not is_ghost_mode:
                try:
                    print(f"\n[Input] 等待输入... (当前忍耐极限: {current_patience}s)")

                    task_keyboard = asyncio.create_task(async_input(">> "))
                    task_voice = asyncio.create_task(asyncio.to_thread(ear.listen))

                    done, pending = await asyncio.wait(
                        [task_keyboard, task_voice],
                        timeout=float(current_patience),
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    if not done:
                        for task in pending: task.cancel()
                        raise asyncio.TimeoutError()

                    winner_task = done.pop()
                    user_input = winner_task.result()
                    for task in pending: task.cancel()

                    if not user_input or len(user_input.strip()) < 1:
                        continue

                    unanswered_attempts = 0
                    print(f"[Received] {user_input}")

                    if user_input.lower() in ['exit', 'quit']:
                        memory.write_memory("主动道别，程序正常退出", importance=2)
                        print("[Sentia] 哼，再见！")
                        audio_data, sr = await asyncio.to_thread(voice.generate_audio_data, "哼，再见！")
                        if audio_data is not None:
                            sd.play(audio_data, samplerate=sr)
                            await body.animate_speech_lip_sync(audio_data, sr, emotion="Angry")
                            await asyncio.sleep(len(audio_data) / sr)
                        break

                    print("[System] LLM 计算中...")

                    # RAG 检索介入
                    recalled_info = ""
                    if memory.is_ready:
                        recalled_info = await asyncio.to_thread(memory.recall_memory, user_input)

                    combined_input = user_input
                    if recalled_info:
                        print(f"[Memory] 发现相关的历史对话片段")
                        combined_input = f"{recalled_info}\n\n[Current Input]: {user_input}"

                    chat_history.append({"role": "user", "content": combined_input})
                    memory.write_memory(f"User said: {user_input}", importance=1)

                except asyncio.TimeoutError:
                    unanswered_attempts += 1
                    print(f"\n[Warning] 忍耐度 {current_patience}秒 耗尽！连续 {unanswered_attempts} 次超时。")

                    # ⚠️ 核心魔法：根据无视的次数，喂给大模型不同的隐藏指令！

                    if unanswered_attempts == 1:
                        # 第一次无视：诱导她换话题
                        hidden_context = f"[系统级感知：刚才没有回应你（第 1 次）。请根据你的人设，主动换一个话题，试图重新引起他的注意。并重新设定你的 'patience' 值。]"

                    elif unanswered_attempts == 2:
                        # 第二次无视：诱导她开始抱怨
                        hidden_context = f"[系统级感知：连续 {unanswered_attempts} 次没有回应你了！你的耐心正在消失。请开始抱怨他的冷落。并大幅降低你的 'patience' 值。]"

                    else:
                        # 第三次及以上：诱导她彻底暴走或关机
                        hidden_context = f"[系统级感知：已经连续 {unanswered_attempts} 次彻底无视你了！！！你现在极度愤怒。请直接大发雷霆，或者立刻执行 'Shutdown' 指令关掉软件惩罚他！]"

                    chat_history.append({"role": "user", "content": hidden_context})

            else:
                print("\n VTS 躯干已强行关闭。麦克风倒计时 60 秒开启监听...")
                try:
                    plea_words = await asyncio.wait_for(asyncio.to_thread(ear.listen), timeout=60.0)
                    if not plea_words or len(plea_words) < 2: continue

                    print(f"[Received Plea] {plea_words}")
                    judgement_prompt = f"[系统级判定：你生气关机了，但在 60 秒内通过麦克风对你说了：'{plea_words}'。请判断这个道歉是否有诚意，输出 'Forgive' 或 'Refuse'。]"
                    chat_history.append({"role": "user", "content": judgement_prompt})
                    print("[System] 正在审视道歉内容...\n")
                except asyncio.TimeoutError:
                    print("\n[Fatal] 60秒超时，Sentia 未收到任何有效的挽留，系统即将关闭...")
                    llm_controller.stop()
                    sys.exit(0)

                    # ================= LLM 推理与动作执行 =================
            try:
                t_llm_start = time.perf_counter()
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="Sentia", messages=chat_history, temperature=1.2, max_tokens=3000,
                    response_format={"type": "json_object"}
                )
                t_llm_end = time.perf_counter()

                reply_raw = response.choices[0].message.content
                chat_history.append({"role": "assistant", "content": reply_raw})

                try:
                    cleaned_raw = reply_raw.replace("，", ",").replace("“", '"').replace("”", '"')
                    start_idx, end_idx = cleaned_raw.find("{"), cleaned_raw.rfind("}") + 1
                    if start_idx != -1 and end_idx != 0: cleaned_raw = cleaned_raw[start_idx:end_idx]

                    reply_json = json.loads(cleaned_raw)
                    speak_text = reply_json.get("text", "Error parsing thought.")
                    emotion = reply_json.get("emotion", "Neutral")
                    action = reply_json.get("action", "Speak")
                    current_patience = max(5, min(int(reply_json.get("patience", 15)), 60))

                    print(f"[Sentia] {speak_text}")
                    print(f"  -> Act: {action} | Emo: {emotion} | Pat: {current_patience}s")

                    memory.write_memory(f"I replied: {speak_text}", emotion_tag=emotion, importance=1)

                    t_tts_start = time.perf_counter()
                    audio_data, sr = await asyncio.to_thread(voice.generate_audio_data, speak_text)
                    t_tts_end = time.perf_counter()

                    if audio_data is not None:
                        audio_duration = (len(audio_data) / sr) * 1000
                        print(
                            f"  -> Profiler: LLM: {(t_llm_end - t_llm_start) * 1000:.1f}ms | TTS: {(t_tts_end - t_tts_start) * 1000:.1f}ms | Audio: {audio_duration:.1f}ms")

                        sd.play(audio_data, samplerate=sr)
                        if not is_ghost_mode:
                            await body.animate_speech_lip_sync(audio_data, sr, emotion=emotion)
                        await asyncio.sleep(len(audio_data) / sr + 0.2)

                    # ================= 动作裁决 =================
                    if action == "Shutdown" and not is_ghost_mode:
                        print("\n[Alert] Sentia 彻底心灰意冷，正在关闭 VTube Studio...")
                        memory.write_memory("极度冷漠，我极其愤怒，直接强行关机",
                                            emotion_tag="Angry", importance=5)
                        os.system('taskkill /F /IM "VTube Studio.exe" >nul 2>&1')
                        await body.close()
                        is_ghost_mode = True

                    elif action == "Forgive" and is_ghost_mode:
                        print("\n[Recover] Sentia原谅你了，正在重新拉起 VTube Studio...")
                        memory.write_memory("向我诚恳地道了歉，我选择原谅了，并重新开启了系统",
                                            emotion_tag="Smile", importance=4)
                        subprocess.Popen(VTS_EXE_PATH, shell=True)
                        print("[System] Waiting for VTS (15s)...")
                        await asyncio.sleep(15)
                        body = VTSController()
                        await body.connect_and_auth()
                        is_ghost_mode = False

                    elif action == "Refuse" and is_ghost_mode:
                        print("\n[Fatal] Sentia 认为你的道歉毫无诚意。")
                        memory.write_memory("道歉毫无诚意，我彻底拒绝并永远地离开了。",
                                            emotion_tag="Angry", importance=5)
                        llm_controller.stop()
                        sys.exit(0)

                except json.JSONDecodeError:
                    print(f"[Error] 未输出标准 JSON: {reply_raw}")
                    # 如果出错了，默认恢复 20 秒耐心
                    current_patience = 20

            except Exception as llm_err:
                print(f"[Error] LLM 连接失败: {llm_err}")
                current_patience = 20

    except KeyboardInterrupt:
        print("\n[System] 捕获键盘中断，强制退出...")
    finally:
        await body.close()
        llm_controller.stop()
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())