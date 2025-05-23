import requests
import time

def get_weight_from_esp32(esp32_ip):
    try:
        url = f"http://{esp32_ip}/"
        response = requests.get(url, timeout=2)

        if response.status_code == 200:
            weight_str = response.text.strip()
            weight = float(weight_str)
            print(f"✅ Weight received: {weight:.2f} g")
            return weight
        else:
            print(f"⚠️ Unexpected response: {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Error contacting ESP32: {e}")
        return None

# === Live Readings ===
if __name__ == "__main__":
    esp32_ip = "172.20.10.2" 

    print("📟 Reading live weight values from ESP32 (Press Ctrl+C to stop)")
    while True:
        get_weight_from_esp32(esp32_ip)
        time.sleep(1)  # 1 second delay between readings

