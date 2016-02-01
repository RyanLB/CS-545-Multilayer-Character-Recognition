//
//  main.swift
//  ml_chars
//
//  Created by Ryan Bernstein on 1/23/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation

// Hardcoding these for now because why not
let trainingDataPath = "/Users/rlb/Documents/School (Current)/CS 545 Machine Learning/Multilayer Character Recognition/training_data.csv"
let testDataPath = "/Users/rlb/Documents/School (Current)/CS 545 Machine Learning/Multilayer Character Recognition/test_data.csv"

let outputDirectory = "/Users/rlb/Documents/School (Current)/CS 545 Machine Learning/Multilayer Character Recognition"

let client = MLClient()

print("Loading data...")
try! client.loadTrainingData(trainingDataPath)
try! client.loadTestData(testDataPath)
print("Done.")

print("Scaling data...")
client.scaleData()
print("Done.")

print("Running experiment 1...")
client.testWithParameters(4, learningRate: 0.3, momentum: 0.3, epochLimit: 100).writeToCSV("\(outputDirectory)/exp1.csv")

print("ayy")