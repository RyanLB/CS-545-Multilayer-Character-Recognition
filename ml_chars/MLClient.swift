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
    
    func testWithParameters(hiddenNodes: Int, learningRate: Double, momentum: Double, epochLimit: Int) {
        let network = NeuralNetwork(hiddenLayerWidth: hiddenNodes)
        
        // I'm a bad programmer who ignores exceptions
        let history = try! network.train(trainingData, testData: testData, learningRate: learningRate, momentum: momentum, epochLimit: epochLimit)
        
        
    }
    
    /// Scales training data to have 0 mean and unit variance for each feature.
    func scaleTrainingData() {
        guard trainingData.count > 0 else {
            return
        }
        
        let trainingDataStats = calculateStats(trainingData.map{ $0.attributeVector })
        for l in trainingData {
            let meanDiffs = zip(l.attributeVector, trainingDataStats.means).map{ $0.0 - $0.1 }
            l.attributeVector = zip(meanDiffs, trainingDataStats.standardDeviations).map{ $0.0 / $0.1 }
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