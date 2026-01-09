from pynput import keyboard
import random, time

isRunning = False
controller = keyboard.Controller()

while True:
    fishNum = -1
    try:
        fishNum = int(input("Enter #fishes: "))
    except ValueError:
        print("Please enter a number bigger than 0!")
    if fishNum >= 1:
        break

def on_press(key):
    pass

def on_release(key):
    global isRunning
    if key == keyboard.Key.f8:
        print("Running script...")
        for f in range(fishNum * 3):
            controller.tap(keyboard.Key.space)
            if (fishNum % 3) == 0:
                controller.tap('e')  # Account for "Chance for a bigger fish!" event
                print(f"Caught fish #{f // 3}")
            time.sleep(5.5 + random.random() / 2)


with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
print("Quitting...")
