import pickle
import numpy as np
from PIL import Image
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Color, Line
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button


from core import forward_pass 

# 1. 94.5% 
try:
    with open('trained_e10.pkl', 'rb') as f:
        brain = pickle.load(f)
except FileNotFoundError:
    print("weights not found")

class DrawingCanvas(Widget):
    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 1) # White ink (MNIST style)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=40) # Thick lines!

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]

class TesterApp(App):
    def build(self):
        self.root = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # The Drawing Area
        self.painter = DrawingCanvas()
        
        # The Result Label
        self.result_label = Label(
            text="Draw a digit...", 
            size_hint_y=0.2, 
            font_size='25sp',
            color=(0, 1, 0.8, 1) 
        )
        
        # Buttons
        btn_layout = BoxLayout(size_hint_y=0.2, spacing=10)
        clear_btn = Button(text='Clear Canvas', background_color=(0.8, 0.2, 0.2, 1))
        predict_btn = Button(text='Predict', background_color=(0.2, 0.6, 0.2, 1))
        
        clear_btn.bind(on_release=self.clear_canvas)
        predict_btn.bind(on_release=self.run_prediction)
        
        btn_layout.add_widget(clear_btn)
        btn_layout.add_widget(predict_btn)
        
        self.root.add_widget(self.painter)
        self.root.add_widget(self.result_label)
        self.root.add_widget(btn_layout)
        
        return self.root

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        self.result_label.text = "Canvas cleared! Draw again. "
    def run_prediction(self, obj):
        # Save and Open
        self.painter.export_to_png("digit_capture.png")
        img = Image.open("digit_capture.png").convert('L')
        
        #  Find the digit
        img = img.point(lambda p: p > 50 and 255) 
        bbox = img.getbbox()
        

        #  Process the shape
        digit_only = img.crop(bbox)
        w, h = digit_only.size
        ratio = 20.0 / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        digit_rescaled = digit_only.resize(new_size, Image.Resampling.LANCZOS)
        
      
        final_img = Image.new("L", (28, 28), 0)
        paste_x = (28 - new_size[0]) // 2
        paste_y = (28 - new_size[1]) // 2
        final_img.paste(digit_rescaled, (paste_x, paste_y))
        
       
        final_img.save("debug.png") 

    
        normalized = [p / 255.0 for p in np.array(final_img).flatten()]
        h1, h2, pred = forward_pass(normalized, brain)
        
        winner = pred.index(max(pred))
        confidence = max(pred) * 100
        self.result_label.text = f"it's a {winner} ({confidence:.1f}%)"
        
        
if __name__ == '__main__':
    TesterApp().run()
    