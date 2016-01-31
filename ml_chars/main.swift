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

let client = MLClient()
try! client.loadTrainingData(trainingDataPath)
try! client.loadTestData(testDataPath)

let stats = client.trainingDataStats()
print ("Means: \(stats.means)")
print ("Standard deviations: \(stats.standardDeviations)")

print("ayy")