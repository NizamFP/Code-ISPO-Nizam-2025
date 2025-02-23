#include <MicroTFLite.h>
#include "model.h"  // The generated header file from the TFLite model

#define NUM_CLASSES 22  // Number of crop types

// Define crop names corresponding to each class index
const char* CROP_NAMES[NUM_CLASSES] = {
    "apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton",
    "grapes", "jute", "kidneybeans", "lentil", "maize", "mango", "mothbeans",
    "mungbean", "muskmelon", "orange", "papaya", "pigeonpeas", "pomegranate",
    "rice", "watermelon"
};

constexpr int kTensorArenaSize = 8 * 1024;  // Adjust size based on model requirements
alignas(16) uint8_t tensorArena[kTensorArenaSize];

// Simulated test dataset
float testDataset[][8] = {
    {90, 42, 43, 20.87974371, 82.00274423, 6.502985292, 202.9355362, 20},  // Example input for "rice"
    {71, 54, 16, 22.61359953, 63.69070564, 5.749914421, 87.75953857, 11},  // Example input for "maize"
    // Add more test cases as needed
};

int numTestCases = sizeof(testDataset) / sizeof(testDataset[0]);

// Replace these with actual mean and stddev values from your dataset
float mean[] = {50.5518181818182, 53.3627272727273, 48.1490909090909, 25.6162438517795, 71.4817792177865, 6.46948006525637, 103.463655415768};  // Example means
float stddev[] = {36.9173338337566, 32.9858827385872, 50.6479305466601, 5.06374859995884, 22.2638115897609, 0.773937688029815, 54.9583885248779};  // Example standard deviations

void setup() {
    Serial.begin(115200);
    while (!Serial) ;

    Serial.println("Crop Recommendation Model");

    if (!ModelInit(model, tensorArena, kTensorArenaSize)) {
        Serial.println("Model initialization failed!");
        while (true) ;
    }
    Serial.println("Model initialization done.");
}

void loop() {
    int correctPredictions = 0;

    for (int i = 0; i < numTestCases; ++i) {
        // Normalize input data
        float inputData[7];
        for (int j = 0; j < 7; ++j) {
            inputData[j] = (testDataset[i][j] - mean[j]) / stddev[j];
            if (!ModelSetInput(inputData[j], j)) {
                Serial.println("Failed to set input!");
                return;
            }
        }

        // Run inference
        if (!ModelRunInference()) {
            Serial.println("RunInference Failed!");
            return;
        }

        // Get output
        int bestIndex = 0;
        float bestProbability = 0.0;
        for (int k = 0; k < NUM_CLASSES; ++k) {
            float probability = ModelGetOutput(k);
            if (probability > bestProbability) {
                bestProbability = probability;
                bestIndex = k;
            }
        }

        int expectedIndex = (int)testDataset[i][7];
        if (bestIndex == expectedIndex) {
            correctPredictions++;
        }

        Serial.print("Test Case ");
        Serial.print(i + 1);
        Serial.print(": Expected ");
        Serial.print(CROP_NAMES[expectedIndex]);
        Serial.print(", Predicted ");
        Serial.print(CROP_NAMES[bestIndex]);
        Serial.print(" with probability: ");
        Serial.print(bestProbability * 100, 2);
        Serial.println("%");
    }

    float accuracy = ((float)correctPredictions / numTestCases) * 100.0;
    Serial.print("Model Accuracy: ");
    Serial.println(accuracy, 2);
    Serial.println("%");

    delay(10000);
}