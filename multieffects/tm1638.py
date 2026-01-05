import RPi.GPIO as GPIO # pyright: ignore[reportMissingModuleSource]
import time

class TM1638:
    def __init__(self, dio, clk, stb):
        self.dio = dio
        self.clk = clk
        self.stb = stb
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.stb, GPIO.OUT)
        GPIO.setup(self.clk, GPIO.OUT)
        GPIO.setup(self.dio, GPIO.OUT)
        self._write_command(0x88 | 7) # เปิดหน้าจอ ความสว่างสูงสุด

    def _write_byte(self, b):
        for i in range(8):
            GPIO.output(self.clk, GPIO.LOW)
            GPIO.output(self.dio, (b >> i) & 1)
            GPIO.output(self.clk, GPIO.HIGH)

    def _write_command(self, cmd):
        GPIO.output(self.stb, GPIO.LOW)
        self._write_byte(cmd)
        GPIO.output(self.stb, GPIO.HIGH)

    def set_text(self, text):
        # แปลงตัวอักษรเป็น 7-segment (เบื้องต้น)
        segments = {'0':0x3f,'1':0x06,'2':0x5b,'3':0x4f,'4':0x66,'5':0x6d,'6':0x7d,'7':0x07,'8':0x7f,'9':0x6f,
                    'A':0x77,'b':0x7c,'C':0x39,'d':0x5e,'E':0x79,'F':0x71,'P':0x73,'n':0x54,'o':0x5c,'r':0x50,'t':0x78,' ':0x00,'-':0x40}
        self._write_command(0x40)
        GPIO.output(self.stb, GPIO.LOW)
        self._write_byte(0xC0)
        text = text[:8].ljust(8)
        for char in text:
            self._write_byte(segments.get(char, 0x00))
            self._write_byte(0x00) # ข้าม byte LED
        GPIO.output(self.stb, GPIO.HIGH)

    def set_led(self, pos, state):
        self._write_command(0x44)
        GPIO.output(self.stb, GPIO.LOW)
        self._write_byte(0xC1 + (pos * 2))
        self._write_byte(1 if state else 0)
        GPIO.output(self.stb, GPIO.HIGH)

    def get_keys(self):
        keys = 0
        GPIO.output(self.stb, GPIO.LOW)
        self._write_byte(0x42)
        GPIO.setup(self.dio, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        for i in range(32):
            GPIO.output(self.clk, GPIO.LOW)
            if GPIO.input(self.dio):
                keys |= (1 << i)
            GPIO.output(self.clk, GPIO.HIGH)
        GPIO.setup(self.dio, GPIO.OUT)
        GPIO.output(self.stb, GPIO.HIGH)
        return [(keys >> i) & 1 for i in [0, 8, 16, 24, 1, 9, 17, 25]]
