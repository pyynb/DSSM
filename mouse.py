import pyautogui
import random
import time

def random_mouse_move():
    screen_width, screen_height = pyautogui.size()  # 获取屏幕分辨率
    print(f"Screen size: {screen_width}x{screen_height}")

    try:
        while True:
            # 随机生成新的鼠标位置
            new_x = random.randint(0, screen_width - 1)
            new_y = random.randint(0, screen_height - 1)

            # 移动鼠标到新位置
            pyautogui.moveTo(new_x, new_y, duration=0.2)
            print(f"Moved mouse to: ({new_x}, {new_y})")

            time.sleep(1)  # 每秒移动一次
    except KeyboardInterrupt:
        print("Program stopped by user.")

if __name__ == "__main__":
    random_mouse_move()

