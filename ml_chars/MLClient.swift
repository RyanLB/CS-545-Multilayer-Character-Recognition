//
//  MLClient.swift
//  ml_chars
//
//  Created by Ryan Bernstein on 1/27/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation

enum MLClientError : ErrorType {
    case InvalidFilepath(path: String)
}

class MLClient {
    var trainingData = [Letter]()
    var testData = [Letter]()
    
    /// Wrapper function to load training data from a  file.
    func loadTrainingData(fromPath: String) throws {
        trainingData = try loadLetters(fromPath)
    }
    
    /// Wrapper function to load test data from a file.
    func loadTestData(fromPath: String) throws {
        testData = try loadLetters(fromPath)
    }
    
    /**
     Wrapper function that creates and trains a network, returning its accuracy history over the course of training.
     
     - Param hiddenNodes: The number of nodes in the `NeuralNetwork`'s hidden layer.
     - Param learningRate: The learning rate with which to train the network.
     - Param momentum: The momentum coefficient with which to train the network.
     - Param epochLimit: The maximal number of training epochs to be used in training.
     
     - Returns: The `AccuracyHistory` that results from the training process.
     */
    func testWithParameters(hiddenNodes: Int, learningRate: Double, momentum: Double, epochLimit: Int) -> AccuracyHistory {
        let network = NeuralNetwork(hiddenLayerWidth: hiddenNodes)
        
        // I'm a bad programmer who ignores exceptions
        return try! network.train(trainingData, testData: testData, learningRate: learningRate, momentum: momentum, epochLimit: epochLimit)
    }
    
    /// Wrapper function to scale both training and test data
    func scaleData() {
        if (trainingData.count > 0) {
            scaleData(trainingData)
        }
        
        if (testData.count > 0) {
            scaleData(testData)
        }
    }
    
    /// Scales `Letter` attribute vectors to have 0 mean and unit variance for each feature.
    private func scaleData(data: [Letter]) {
        let stats = calculateStats(data.map{ $0.attributeVector })
        for l in data {
            let meanDiffs = zip(l.attributeVector, stats.means).map{ $0.0 - $0.1 }
            l.attributeVector = zip(meanDiffs, stats.standardDeviations).map{ $0.0 / $0.1 }
        }
    }
    
    
    /**
     Calculates the mean and standard deviation for each column in the given two-dimensional array.
     
     - Param dataset: A two-dimensional array of `Double`s from which to calculate statistics.
     
     - Returns: A tuple with the means and standard deviations for each column as `[Double]`s.
     */
    func calculateStats(dataset: [[Double]]) -> (means: [Double], standardDeviations: [Double]) {
        let attributeCount = dataset[0].count
        
        let means = Vector(data: [Double](count: attributeCount, repeatedValue: 0.0))
        dataset.forEach{
            try! means.add(Vector(data: $0), scale: nil)
        }
        
        means.scale(1 / Double(dataset.count))
        
        let sumSquaredDifferences = Vector(data: [Double](count: attributeCount, repeatedValue: 0.0))
        for l in dataset {
            let diff = Vector(data: l)
            try! diff.add(means, scale: -1.0)
            try! sumSquaredDifferences.add(Vector.hadamardProduct(diff, v2: diff), scale: nil)
        }
        
        sumSquaredDifferences.scale(1.0 / Double(dataset.count))
        
        let standardDeviations = sumSquaredDifferences.data.map{ sqrt($0) }
        
        return (means: means.data, standardDeviations: standardDeviations)
    }
    
    /**
     Loads an array of Letter objects from a file at the given path.
     
     - Parameter fromPath: The relative path of the file from which to load Letters.
     
     - Returns: A list of Letter objects.
     */
    private func loadLetters(fromPath: String) throws -> [Letter] {
        let inputHandle = NSFileHandle(forReadingAtPath: fromPath)
        
        if inputHandle == nil {
            throw MLClientError.InvalidFilepath(path: fromPath)
        }
        
        var results = [Letter]()
        
        repeat {
            if let line = inputHandle?.getASCIILine() {
                let letter = Letter.init(withCommaSeparatedAttributeList: line)
                results.append(letter)
            }
            else {
                break
            }
            
        } while true
        
        inputHandle!.closeFile()
        
        return results
    }
}