from tkinter import * 
import numpy as np
import serial
import time
import struct
import threading

try:
    with serial.Serial("COM3", 500000, timeout = 5) as ser:
        print("Connect Success")
        time.sleep(2)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        latest_snapshot = None
        data_lock = threading.Lock()
        image_change = threading.Event()
        
        window = Tk()
        window_size = 840
        draw_size = 28
        image_size = 14
        image_dp = 14

        brush_radius = 1 # pixels

        window.geometry(f"{ window_size }x{ window_size }")
        window.title("Avrex MNIST Demo")

        canvas = Canvas(window, height = window_size, width = window_size, bg = "Black", highlightthickness = 0)
        canvas.pack()

        image = np.zeros((draw_size, draw_size))
        drawing = False

        result_text = canvas.create_text(20, 20, anchor = "nw", text = "5", font = ("Arial", 36), fill = "red")

        def draw_pixel(x, y, intensity, canvas: Canvas):
            square_size = (window_size / draw_size)
            true_x = x * square_size
            true_y = y * square_size

            intensity = int(intensity * 255)
            canvas.create_rectangle(true_x, true_y, true_x + square_size, true_y + square_size, fill = f"#{intensity:02x}{intensity:02x}{intensity:02x}", outline = "")
            
        to_fixed = lambda x: int(round(x * (2 ** image_dp)))
        to_fixed_np = np.vectorize(to_fixed)

        ready_image_buffer = np.zeros(image_size ** 2)
        result_buffer = None # prediction, time

        def update_gui():
            global result_buffer
            if result_buffer is not None:
                prediction, time = result_buffer
                canvas.itemconfig(result_text, text = f"{ prediction } + { time / 1000 }ms")
                result_buffer = None
            window.after(20, update_gui)

        update_gui()

        def on_draw(input):
            if (not drawing): return
            global ready_image_buffer

            x = int(input.x * draw_size / window_size) 
            y = int(input.y * draw_size/ window_size)

            for i in range(x - brush_radius, x + brush_radius + 1):
                for j in range(y - brush_radius, y + brush_radius + 1):
                    if (i > 27) or (j > 27) or (i < 0) or (j < 0): continue

                    dx = abs(i - x); dy = abs(j - y)

                    intensity = 1.0 / (dx + dy + 1)
                    image[j, i] = max(image[j, i], intensity)
                    draw_pixel(i, j, image[j, i], canvas)

            factor = int(draw_size / image_size)
            ds_image = image.reshape(image_size, factor, image_size, factor).mean(axis = (1, 3))

            if ds_image.max() > 0:
                ds_image = ds_image / ds_image.max()

            fixed_image = np.round(ds_image.flatten() * (1 << image_dp)).astype(np.int16)
            ready_image_buffer = fixed_image

            image_change.set()
            canvas.tag_raise(result_text)

        def cycle_draw_state(input): 
            global drawing
            drawing = not drawing
            if drawing: on_draw(input)

        def clear_image(event):
            global image
            global result_text
            image[:] = 0
            canvas.delete("all")

            result_text = canvas.create_text(20, 20, anchor = "nw", text = "5", font = ("Arial", 36), fill = "red")

        def inference_loop():
            global latest_snapshot
            global result_buffer
            
            while True:
                image_change.wait()
                image_change.clear()
                
                with data_lock: latest_snapshot = ready_image_buffer
                bytes = latest_snapshot.tobytes()
                ser.write(bytes)
                guess = ser.read(5)

                if len(guess) == 5:
                    predicted_digit, time = struct.unpack("<BI", guess)
                    print(predicted_digit)

                    result_buffer = (predicted_digit, time)
                else:
                    print("Model Timed Out")
                
        infer_thread = threading.Thread(target = inference_loop, daemon = True)
        infer_thread.start()

        window.bind("<Button-1>", cycle_draw_state)
        window.bind("<ButtonRelease-1>", cycle_draw_state)
        window.bind("<Motion>", on_draw)
        window.bind("<BackSpace>", clear_image)

        window.mainloop()

except serial.SerialException as exception:
    print(f"Serial failed { exception }")