//
//  NeuralNetwork.swift
//  ml_chars
//
//  Created by Ryan Bernstein on 1/26/16.
//  Copyright © 2016 Ryan Bernstein. All rights reserved.
//

import Foundation
    
class NeuralNetwork {
    private var _outputLayer: OutputLayer
    private var _hiddenLayer: HiddenLayer // Just one for now. May extend with a list later
    
    var outputLayer: OutputLayer {
        get { return _outputLayer }
    }
    
    var hiddenLayer: HiddenLayer {
        get { return _hiddenLayer }
    }
    
    init(hiddenLayerWidth: Int) {
        _outputLayer = OutputLayer(weightCount: 26, inputCount: hiddenLayerWidth)
        _hiddenLayer = HiddenLayer(weightCount: hiddenLayerWidth, inputCount: 16)
    }
    
    /// Copy constructor, wheee
    init(original: NeuralNetwork) {
        _outputLayer = OutputLayer(original: original.outputLayer)
        _hiddenLayer = HiddenLayer(original: original.hiddenLayer)
    }
    
    /**
     Guesses which character a given Letter represents.
     
     - Throws: `VectorError.MismatchedLength` if inputs to some layer don't match the expected length.
     
     - Returns: The character with the highest output activation. Ties will be broken via random number generation,
       but are unlikely.
     */
    func test(l: Letter) throws -> Character {
        //let result = try outputLayer.calculateOutput(hiddenLayer.calculateOutput(Vector(data: l.attributeVector))).data
        let hiddenOutput = try hiddenLayer.calculateOutput(Vector(data: l.attributeVector))
        let result = try outputLayer.calculateOutput(hiddenOutput).data
        return guessFromActivations(result)
    }
    
    /**
     Guesses which character a given Letter represents, providing activation information for training.
     
     - Throws: `VectorError.MismatchedLength` if inputs to some layer don't match the expected length.
     
     - Returns: A tuple containing activations from the hidden layer, activations from the output layer, and the
       character with the highest activation. Ties will be broken via random number generation, but are unlikely.
     */
    private func testWithLayerActivations(l: Letter) throws -> (hiddenActivations: Vector, outputActivations: Vector, guess: Character) {
        let hiddenActivations = try hiddenLayer.calculateOutput(Vector(data: l.attributeVector))
        let outputActivations = try outputLayer.calculateOutput(hiddenActivations)
        let guess = guessFromActivations(outputActivations.data)
        
        return (hiddenActivations: hiddenActivations, outputActivations: outputActivations, guess: guess)
    }
    
    /**
     Trains this network on a given training set.
     
     - Param trainingData: A list of `Letter`s on which to train.
     - Param testData: The test set. Accuracy is calculated over this set, but it is not used in training.
     - Param learningRate: The learning rate at which we modify weights.
     - Param momentum: The rate at which we update weights based on past updates.
     - Param epochLimit: The number of epochs for which to train this network.
     
     - Returns: The history of training and test accuracies over the course of training.
     */
    func train(trainingData: [Letter], testData: [Letter], learningRate: Double, momentum: Double, epochLimit: Int) throws -> AccuracyHistory {
        // There's definitely a cleaner way to structure this, but it was originally written with the assumption that
        // we would stop training when accuracy stopped increasing, as we did in our Perceptron assignment.
        
        let history = AccuracyHistory()
        var trainingAccuracy = try accuracy(trainingData)
        history.add(AccuracyHistory.AccuracyPair(trainingAccuracy: trainingAccuracy, testAccuracy: try accuracy(testData)))
        
        for _ in 0..<epochLimit {
            var shuffled = trainingData
            shuffled.shuffle()
            let newNetwork = try epochResult(shuffled, learningRate: learningRate, momentum: momentum)
            
            let newTrainingAccuracy = try newNetwork.accuracy(trainingData)
            history.add(AccuracyHistory.AccuracyPair(trainingAccuracy: newTrainingAccuracy, testAccuracy: try newNetwork.accuracy(testData)))
            
            if newTrainingAccuracy <= trainingAccuracy {
                //break
            }
            
            trainingAccuracy = newTrainingAccuracy
            
            self._hiddenLayer = newNetwork.hiddenLayer
            self._outputLayer = newNetwork.outputLayer

        }
        
        return history
    }
    
    /// Calculates accuracy (correctly guessed instances / total) over the given dataset.
    private func accuracy(dataset: [Letter]) throws -> Double {
        return try Double(dataset.countWhere{ try self.test($0) == $0.knownValue }) / Double(dataset.count)
    }
    
    /// Creates and returns a new `NeuralNetwork` with the weights that result from a single epoch.
    private func epochResult(trainingData: [Letter], learningRate: Double, momentum: Double) throws -> NeuralNetwork {
        let newNetwork = NeuralNetwork(original: self)
        
        var outputDeltas: Deltas? = nil
        var hiddenDeltas: Deltas? = nil
        
        try trainingData.forEach {
            let results = try newNetwork.testWithLayerActivations($0)
            
            if results.guess != $0.knownValue {
                // Factor this into a method?
                
                var targets = [Double](count: 26, repeatedValue: 0.1)
                targets[NeuralNetwork.AZIndexFromCharacter($0.knownValue)] = 0.9
                
                let outputErrors = newNetwork.outputLayer.calculateErrors(results.outputActivations, targets: Vector(data: targets))
                let weighted = try newNetwork.outputLayer.weightedErrors(outputErrors)
                let hiddenErrors = newNetwork.hiddenLayer.calculateErrors(results.hiddenActivations, nextLayerWeightedErrors: weighted.weightErrors)
                
                outputDeltas = try newNetwork.outputLayer.train(learningRate,
                                                                inputs: results.hiddenActivations,
                                                                errors: outputErrors,
                                                                momentum: momentum,
                                                                previousDeltas: outputDeltas)
                
                hiddenDeltas = try newNetwork.hiddenLayer.train(learningRate,
                                                                inputs: Vector(data: $0.attributeVector),
                                                                errors: hiddenErrors,
                                                                momentum: momentum,
                                                                previousDeltas: hiddenDeltas)
                
            }
        }
        
        return newNetwork
    }
    
    /// Takes 26 activations from the output layer and returns the Character with the highest value.
    private func guessFromActivations(activations: [Double]) -> Character {
        let winners = activations.indicesWhere{ $0 == activations.maxElement() }
        let winnerIndex = Int(arc4random_uniform(UInt32(winners.count)))
        let guess = NeuralNetwork.characterFromAZIndex(winners[winnerIndex])
        return guess
    }
    
    /// Takes an index between 0 and 25 and returns a character between 'A' and 'Z'
    private class func characterFromAZIndex(index: Int) -> Character {
        assert(index >= 0 && index < 26, "Index must be between 0 and 26")
        return Character(UnicodeScalar(index + 65))
    }
    
    /// Takes a character, presumably between 'A' and 'Z', and returns an ordinal from 0-25.
    private class func AZIndexFromCharacter(c: Character) -> Int {
        let s = String(c).unicodeScalars
        let val = Int(s[s.startIndex].value)
        return val - 65
    }
}