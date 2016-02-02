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

// Experiments!
let epochLimit = 100


print("Running experiment 1...")
client.testWithParameters(4, learningRate: 0.3, momentum: 0.3, epochLimit: epochLimit).writeToCSV("\(outputDirectory)/exp1.csv")
print("Done. Running experiment 2a...")
client.testWithParameters(4, learningRate: 0.05, momentum: 0.3, epochLimit: epochLimit).writeToCSV("\(outputDirectory)/exp2a.csv")
print("Done. Running experiment 2b...")
client.testWithParameters(4, learningRate: 0.6, momentum: 0.3, epochLimit: epochLimit).writeToCSV("\(outputDirectory)/exp2b.csv")
print("Done. Running experiment 3a...")
client.testWithParameters(4, learningRate: 0.3, momentum: 0.05, epochLimit: epochLimit).writeToCSV("\(outputDirectory)/exp3a.csv")
print("Done. Running experiment 3b...")
client.testWithParameters(4, learningRate: 0.3, momentum: 0.6, epochLimit: epochLimit).writeToCSV("\(outputDirectory)/exp3b.csv")
print("Done. Running experiment 4a...")
client.testWithParameters(2, learningRate: 0.3, momentum: 0.3, epochLimit: epochLimit).writeToCSV("\(outputDirectory)/exp4a.csv")
print("Done. Running experiment 4b...")
client.testWithParameters(8, learningRate: 0.3, momentum: 0.3, epochLimit: epochLimit).writeToCSV("\(outputDirectory)/exp4b.csv")

print("Done.")