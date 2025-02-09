
/* Get this code from Blynk's dashboard
#define BLYNK_TEMPLATE_ID           "TMPL6Jwp3nNpe"
#define BLYNK_TEMPLATE_NAME         "ISPO"
#define BLYNK_AUTH_TOKEN            "rV98KC5X3fSip9v-gp8VLr0kFCHCLA2Y"

#define BLYNK_PRINT Serial
*/

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>


#include <OneWire.h>
#include <MicroTFLite.h>
#include "model.h"  // The generated header file from the TFLite model
#define NUM_CLASSES 22  // Number of crop types
#include <Adafruit_SSD1306.h> // Memanggil Library OLED SSD1306
#include <DallasTemperature.h>
const int relay = 5;
#define ONE_WIRE_BUS 2
OneWire oneWire(ONE_WIRE_BUS);
#include <DHT.h>
#include <LiquidCrystal_I2C.h>
#define DHT11_PIN 14
LiquidCrystal_I2C lcd(0x27, 16, 2);
DHT dht11(DHT11_PIN, DHT11);
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#define DMSpin  13     // pin output untuk DMS
#define indikator  2   // pin led indikator built-in ESP32
#define adcPin 34      // pin input sensor pH tanah
#define SCREEN_WIDTH 128 // Lebar Oled dalam Pixel
#define SCREEN_HEIGHT 32 // Tinggi Oled dalam Pixel
#define OLED_RESET     4 
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

const char* CROP_NAMES[NUM_CLASSES] = {
    "apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton",
    "grapes", "jute", "kidneybeans", "lentil", "maize", "mango", "mothbeans",
    "mungbean", "muskmelon", "orange", "papaya", "pigeonpeas", "pomegranate",
    "rice", "watermelon"
};
int ADC;
float lastReading;
float pH;
const int AirValue = 1023;   //you need to replace this value with Value_1
const int WaterValue = 440;  //you need to replace this value with Value_2
const int SensorPin = 15;
int soilMoistureValue = 0;
int soilmoisturepercent=0;
int _moisture,sensor_analog;
const int sensor_pin = A0;  /* Soil moisture sensor O/P pin */
int siram=0;
float testDataset[8];
constexpr int kTensorArenaSize = 8 * 1024;  // Adjust size based on model requirements
alignas(16) uint8_t tensorArena[kTensorArenaSize];
int numTestCases = sizeof(testDataset) / sizeof(testDataset[0]);
float mean[] = {50.5518181818182, 53.3627272727273, 48.1490909090909, 25.6162438517795, 71.4817792177865, 6.46948006525637, 103.463655415768};  // Example means
float stddev[] = {36.9173338337566, 32.9858827385872, 50.6479305466601, 5.06374859995884, 22.2638115897609, 0.773937688029815, 54.9583885248779};  // Example standard deviations

// Your WiFi credentials.
// Set password to "" for open networks.
// BLYNKDEACT (DELETE THIS STAR WHEN TESTING BLYNK, THIS IS UNACTIVATED TO SAVE MESSAGES)

char ssid[] = "1234";
char pass[] = "nfp12347";
BlynkTimer timer;

// This function is called every time the Virtual Pin 0 state changes
BLYNK_WRITE(V0)
{
  // Set incoming value from pin V0 to a variable
  int value = param.asInt();

  // Update state
  Blynk.virtualWrite(V1, value);
}

// This function is called every time the device is connected to the Blynk.Cloud
BLYNK_CONNECTED()
{
  // Change Web Link Button message to "Congratulations!"
  Blynk.setProperty(V3, "offImageUrl", "https://static-image.nyc3.cdn.digitaloceanspaces.com/general/fte/congratulations.png");
  Blynk.setProperty(V3, "onImageUrl",  "https://static-image.nyc3.cdn.digitaloceanspaces.com/general/fte/congratulations_pressed.png");
  Blynk.setProperty(V3, "url", "https://docs.blynk.io/en/getting-started/what-do-i-need-to-blynk/how-quickstart-device-was-made");
}

// This function sends Arduino's uptime every second to Virtual Pin 2.
void myTimerEvent()
{
  // You can send any value at any time.
  // Please don't send more that 10 values per second.
  Blynk.virtualWrite(V2, millis() / 1000);
}

//BLYNKDEACT
void setup(){
  // Debug console
  Serial.begin(115200);
  testDataset[0] = 40; //Value N
  testDataset[1] = 10; //Value P
  testDataset[2] = 20; // Value K
  testDataset[6] = 214; // Value Rainfall

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { // alamat I2C 0x3C untuk 128x32
  Serial.println(F("SSD1306 Gagal"));
    for(;;); // mengulang terus, hingga bisa menghubungkan ke I2C Oled
  }
  Serial.println("Crop Recommendation Model");

    if (!ModelInit(model, tensorArena, kTensorArenaSize)) {
        Serial.println("Model initialization failed!");
        while (true) ;
    }
    Serial.println("Model initialization done.");

  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);
  // You can also specify server:
  //Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass, "blynk.cloud", 80);
  //Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass, IPAddress(192,168,1,100), 8080);

  // Setup a function to be called every second
  // BLYNKDEACT
  timer.setInterval(1000L, myTimerEvent);
  dht11.begin();
  lcd.begin();
  lcd.backlight();
  analogReadResolution(10);      // setting resolusi pembacaan ADC menjadi 10 bit
  pinMode(DMSpin, OUTPUT);
  pinMode(indikator, OUTPUT);
  digitalWrite(DMSpin,HIGH);     // non-aktifkan DMS
  pinMode(relay, OUTPUT);

}

unsigned long previousMillis = 0; // Declare previousMillis at the top
const long interval = 3000;       // Set your desired interval (in milliseconds)

void loop() {
  
  digitalWrite(DMSpin,LOW);      // aktifkan DMS
  digitalWrite(indikator, HIGH); // led indikator built-in ESP32 menyala
  delay(1*500);                // wait DMS capture data
  ADC = analogRead(adcPin); 

  display.clearDisplay();                 //Membersihkan tampilan
  display.setTextSize(1);                 //Ukuran tulisan
  display.setTextColor(SSD1306_WHITE);
  // BLYNKDEACT
  
  Blynk.run();
  timer.run();
  
  float humi = dht11.readHumidity();
  float tempC = dht11.readTemperature();
  if (isnan(humi) || isnan(tempC)) {
    Serial.println("Semsor tidak terbaca!");
  }
  //Serial.print("Temperature: ");
  Serial.print("temperature: ");
  Serial.println(tempC);
  //Serial.println(" °C");

  display.setCursor(0,0);                 // Koordinat awal tulisan (x,y) dimulai dari atas-kiri
  display.print(F("Temp: "));
  display.print(tempC, 1);  
  display.print("C");

  //Serial.print("Humidity: ");
  Serial.print("humidity: ");
  Serial.println(humi, 0);
  //Serial.println("%");

  display.setCursor(74,0);                 // Koordinat awal tulisan (x,y) dimulai dari atas-kiri
  display.print(F("Humi: "));       
  display.print(humi, 0);  
  display.print(F("%"));

  soilMoistureValue = analogRead(sensor_pin);

  //soilMoistureValue = analogRead(SensorPin);  //put Sensor insert into soil
  Serial.print("moisturevalue: ");
  Serial.println(soilMoistureValue);
  soilmoisturepercent = map(soilMoistureValue, AirValue, WaterValue, 0, 100);

  if(soilmoisturepercent > 100) {
    Serial.print("Moisture: ");
    Serial.println("100 %");
    display.setCursor(0,8);                 // Koordinat awal tulisan (x,y) dimulai dari atas-kiri
    display.print(F("Mois: "));       
    display.print("100");  
    display.print(F("%"));
  }

  else if(soilmoisturepercent <0) {
    Serial.print("Moisture: ");
    Serial.println("0%");
    display.setCursor(0,8);                 // Koordinat awal tulisan (x,y) dimulai dari atas-kiri
    display.print(F("Mois: "));       
    display.print("0");  
    display.print(F("%"));
  }
  else if (soilmoisturepercent >=0 && soilmoisturepercent <= 100) {
    Serial.print("moisture: ");
    Serial.println(soilmoisturepercent);
    //Serial.println("%");
    display.setCursor(0,8);                 // Koordinat awal tulisan (x,y) dimulai dari atas-kiri
    display.print(F("Mois: "));       
    display.print(soilmoisturepercent);  
    display.print(F("%"));
    lcd.setCursor(8, 0);
  }
  

/***************************************************************
 *  Setelah anda melakukan proses kalibrasi dengan benar 
 *  dan mencari rumus kalibrasi menggunakan regresi linier excel
 *  maka nilai rumus regresi dibawah ini gantilah dengan nilai
 *  rumus regresi yang anda dapatkan
 ***************************************************************/
  
  pH = (-0.0233 * ADC) + 12.698;  // ini adalah rumus regresi linier yang wajib anda ganti!
    if (pH != lastReading) { 
    lastReading = pH; 
    }
  Serial.print("ADC: ");
  Serial.println(ADC);             // menampilkan nilai ADC di serial monitor pada baudrate 115200
  Serial.print("pH: ");
  Serial.println(lastReading, 1); // menampilkan nilai pH di serial monitor pada baudrate 115200

  display.setCursor(0,16);                 // Koordinat awal tulisan (x,y) dimulai dari atas-kiri
  display.print(F("pH: "));       
  display.print(lastReading,1);  

  digitalWrite(DMSpin,HIGH);
  digitalWrite(indikator,LOW);  

  
  
  int correctPredictions = 0;
  float inputData[7];
  for (int j = 0; j < 7; ++j) {
    inputData[j] = (testDataset[j] - mean[j]) / stddev[j];
    if (!ModelSetInput(inputData[j], j)) {
        Serial.println("Failed to set input!");
        return;
    }
  }
  if (!ModelRunInference()) {
    Serial.println("RunInference Failed!");
    return;
  }
  testDataset[3] = tempC;
  testDataset[4] = humi;
  testDataset[5] = lastReading;

  int bestIndex = 0;
  float bestProbability = 0.0;
  for (int k = 0; k < NUM_CLASSES; ++k) {
    float probability = ModelGetOutput(k);
    if (probability > bestProbability) {
        bestProbability = probability;
        bestIndex = k;
    }
  }

  int expectedIndex = (int)testDataset[7];
  

  Serial.println(CROP_NAMES[bestIndex]);
  //Serial.print(", with probability: ");
  Serial.print("confidence: ");
  Serial.println(bestProbability * 100, 2);
  //Serial.println("_____________________________________");
  display.setCursor(0,24);                 // Koordinat awal tulisan (x,y) dimulai dari atas-kiri     
  display.print(CROP_NAMES[bestIndex]);
  display.print(", ");
  display.print(bestProbability * 100, 2);
  display.display();
  Blynk.virtualWrite(V10, tempC);
  Blynk.virtualWrite(V11, humi);
  Blynk.virtualWrite(V12, soilmoisturepercent);
  Blynk.virtualWrite(V13, lastReading);
  Blynk.virtualWrite(V14, CROP_NAMES[bestIndex]);
  Blynk.virtualWrite(V15, bestProbability * 100);

  delay(5000);
  
}

