package ml.classifiers;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import java.util.*;

public class TwoLayerNN implements Classifier {

	private int numHidden;
	private double eta = 0.1; // learning rate for the network
	private int numIterations = 200; // number of times to iterate over training
										// data
	private ArrayList<ArrayList<Double>> hiddenWeights;
	private ArrayList<Double> layerTwoWeights;
	private Random random; // random double generator to initialize hidden
							// weights
	private ArrayList<ArrayList<Double>> hiddenOutputs;

	/**
	 * Constructor for a two-layer neural network
	 * 
	 * @param nodes
	 *            number of hidden nodes in the network
	 */
	public TwoLayerNN(int nodes) {
		numHidden = nodes;
		hiddenWeights = new ArrayList<ArrayList<Double>>();
		layerTwoWeights = new ArrayList<Double>();
		random = new Random();
		hiddenOutputs = new ArrayList<ArrayList<Double>>();

	}

	// 1. calculate output of all nodes
	// need 2 weight vectors: 1 of dimensions m*d (b/w weights and hidden)
	// and 1 single dimension of length d (b/w hidden & output layer)
	// f'x = 1-(tanh)^2;
	// Math.tanh(double);

	// 2. calculate updates directly for output layer
	// 3. backpropagate through hidden layers

	/**
	 * 
	 * 
	 * @param data
	 *            Data set to train upon
	 */
	@Override
	public void train(DataSet data) {
		ArrayList<Example> examples = data.getCopyWithBias().getData();
		// initialize both weight vectors with random values between -0.1 ad 0.1
		initWeights(examples.size());

		// store outputs from forward calculation for each example
		// array will have size # examples.
		ArrayList<Double> nodeOutputs = calculateForward(examples);

		// now take error and back-propagate from output to hidden nodes

		for (int j = 0; j < examples.size(); j++) {
			ArrayList<Double> h = hiddenOutputs.get(j); // get hidden outputs
														// for single example
			Example ex = examples.get(j); 
			double vDotH = 0.0;

			for (int i = 0; i < layerTwoWeights.size(); i++) {
				double oldV = layerTwoWeights.get(i);
				double hk = h.get(i);
				vDotH += hk*oldV;
				// v_k = v_k + eta*h_k(y-f(v dot h)) f'(v dot h)
				//need to calculate v dot h?
			}
			for (int i = 0; i < layerTwoWeights.size(); i++) {
				double oldV = layerTwoWeights.get(i);
				double hk = h.get(i);
				vDotH += hk*oldV;
				// v_k = v_k + eta*h_k(y-f(v dot h)) f'(v dot h)
				//need to calculate v dot h?
				layerTwoWeights.set(i, oldV + eta*hk*(ex.getLabel()-Math.tanh(vDotH))*(1-Math.tanh(vDotH)));
			}

		}

		// take that error and back-propagate one more time
		
	}
	
	public double vDotH(ArrayList<Double> outputs, ArrayList<Double> weights) {
		double toReturn = 0.0;
		for (int i = 0; i < weights.size(); i++) {
			double oldV = weights.get(i);
			//double hk = h.get(i);
			//toReturn += hk*oldV;
			// v_k = v_k + eta*h_k(y-f(v dot h)) f'(v dot h)
			//need to calculate v dot h?
			
		}
		return 0.0;
	}

	/**
	 * Method that initializes all weights to values between -.1 and .1
	 * 
	 * @param dataSetSize
	 *            size of the data set
	 */
	public void initWeights(int dataSetSize) {
		// initialize both weight vectors with random values between -0.1 ad 0.1
		for (int i = 0; i < numHidden; i++) {
			ArrayList<Double> temp = new ArrayList<Double>();
			for (int j = 0; j < dataSetSize; j++) {
				temp.add(-.1 + (.1 - (-.1)) * random.nextDouble());
			}
			hiddenWeights.add(temp);
			layerTwoWeights.add(-.1 + (.1 - (-.1)) * random.nextDouble());
		}
	}

	/**
	 * Method which calculates in the forward direction the output of the
	 * network.
	 * 
	 * @param data
	 *            Examples for which an output node value will be calculated.
	 * @return List of network output node values for each respective example in
	 *         the dataset.
	 */
	public ArrayList<Double> calculateForward(ArrayList<Example> data) {
		// store outputs from hidden layer for each example
		// array will have size # examples.
		ArrayList<Double> nodeOutputs = new ArrayList<Double>();

		// randomly shuffle training data
		Collections.shuffle(data);

		// go through each example and calculate forwards
		for (Example ex : data) {
			// get associated weight in hiddenWeights & calculate dot product,
			// run through tanh function
			double bias = ex.getFeature(ex.getFeatureSet().size() - 1);
			ArrayList<Double> thisNodeOutputs = new ArrayList<Double>();
			// loop through features, and the weight vector for each feature
			for (int n = 0; n < numHidden; n++) {
				double thisNode = 0.0; // dot product f&h for each node
				ArrayList<Double> weightVec = hiddenWeights.get(n);
				Set<Integer> features = ex.getFeatureSet();
				Iterator<Integer> iter = features.iterator();
				for (int i = 0; i < features.size(); i++) {
					double featureVal = ex.getFeature(iter.next());
					double weight = weightVec.get(i);
					thisNode += weight * featureVal;
				}
				thisNodeOutputs.add(Math.tanh(thisNode * bias));
			}
			hiddenOutputs.add(thisNodeOutputs);
			// now do same for output layer
			double outSum = 0.0;
			for (int i = 0; i < numHidden; i++) {
				double w = layerTwoWeights.get(i);
				double output = thisNodeOutputs.get(i);
				outSum += w * output;
			}
			nodeOutputs.add(Math.tanh(outSum * bias));
		}
		return nodeOutputs;
	}

	@Override
	public double classify(Example example) {
		// run forwards part of training algorithm on specific example
		return 0;
	}

	@Override
	public double confidence(Example example) {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * Set the learning rate for the network.
	 * 
	 * @param newEta
	 *            new value for eta
	 */
	public void setEta(double newEta) {
		eta = newEta;
	}

	/**
	 * Set number of times to iterate over training data during training.
	 * 
	 * @param iter
	 *            new number of iterations
	 */
	public void setIterations(int iter) {
		numIterations = iter;
	}

	public static void main(String[] args) {
		TwoLayerNN c = new TwoLayerNN(3);
		String file = "/Users/maddie/Documents/FALL2016/MachineLearning/hw4/titanic-train.perc.csv";
		DataSet data = new DataSet(file, 0);
		CrossValidationSet cs = new CrossValidationSet(data, 10, true);

		for (int i = 0; i < cs.getNumSplits(); i++) {
			DataSetSplit dss = cs.getValidationSet(i);
			c.train(dss.getTrain());
			double acc = 0.0;
			double size = dss.getTest().getData().size();
			for (Example ex : dss.getTest().getData()) {
				if (c.classify(ex) == ex.getLabel()) {
					acc += 1.0;
				}
			}
			System.out.println(acc / size);
		}
	}

}
