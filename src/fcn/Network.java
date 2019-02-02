package fcn;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;

import trainset.TrainSet;

public class Network {

    private double[][] outputMatrix;
    private double[][][] weightMatrix; // layer/neuron/previous neuron
    private double[][] biasMatrix; // layer/neuron

    private double[][] errorSignalMatrix;
    private double[][] outputDerivativeMatrix;

    public final int[] LAYER_SIZES;
    public final int INPUT_LAYER_SIZE;
    public final int OUTPUT_LAYER_SIZE;
    public final int TOTAL_NUMBER_OF_LAYERS;

    public Network(int... LAYER_SIZES) throws Exception {
	this.LAYER_SIZES = LAYER_SIZES;
	this.INPUT_LAYER_SIZE = LAYER_SIZES[0];
	this.TOTAL_NUMBER_OF_LAYERS = LAYER_SIZES.length;
	this.OUTPUT_LAYER_SIZE = LAYER_SIZES[TOTAL_NUMBER_OF_LAYERS - 1];

	this.outputMatrix = new double[TOTAL_NUMBER_OF_LAYERS][];
	this.weightMatrix = new double[TOTAL_NUMBER_OF_LAYERS][][];
	this.biasMatrix = new double[TOTAL_NUMBER_OF_LAYERS][];

	this.errorSignalMatrix = new double[TOTAL_NUMBER_OF_LAYERS][];
	this.outputDerivativeMatrix = new double[TOTAL_NUMBER_OF_LAYERS][];

	for (int i = 0; i < TOTAL_NUMBER_OF_LAYERS; i++) {
	    this.outputMatrix[i] = new double[LAYER_SIZES[i]];
	    this.errorSignalMatrix[i] = new double[LAYER_SIZES[i]];
	    this.outputDerivativeMatrix[i] = new double[LAYER_SIZES[i]];

	    this.biasMatrix[i] = Tools.createRandomArray(LAYER_SIZES[i], 0.3, 0.7);

	    if (i > 0) {
		// wagi dla kazdej warstwy oprocz 1.
		weightMatrix[i] = Tools.generateRandomArray(LAYER_SIZES[i], LAYER_SIZES[i - 1], -0.5, 0.5);
	    }
	}
    }
    
    

    /*
     * [A]x[B]= C, Macierz A sklada sie z wag polaczen neuronow, MAcierz B sklada
     * sie z OUTPUTów poprzednich neuronów, macierz C to outputy następnych neuronów
     * Operacje trzeba powtarzać co warstwę
     * 
     */
    public double[] calculateOutputMatrix(double... input) throws Exception {
	if (input.length != this.INPUT_LAYER_SIZE)
	    throw new Exception("Nie zgadza się liczba danych wejściowych i liczba neuronów wejściowych");

	// wyjscie z layer 0 jest rowne jej wejsciu
	this.outputMatrix[0] = input;

	// zaczynam pomijajac pierwsza warstwe; iteruje przez HIDDEN
	for (int layer = 1; layer < TOTAL_NUMBER_OF_LAYERS; layer++) {
	    // iterowanie przez kazdy neuron w warstwie
	    for (int neuron = 0; neuron < LAYER_SIZES[layer]; neuron++) {

		// suma w neuronie to napoczatku sam BIAS dla tego neuronu
		double sum = biasMatrix[layer][neuron];
		// iterowanie przez neurony poprzedniej warstwy
		for (int prevNeuron = 0; prevNeuron < LAYER_SIZES[layer - 1]; prevNeuron++) {
		    // do sumy AKTUALNEGO neuronu trzeba dodać output*waga ze
		    // wszystkich poprzednich polaczonych neuronow
		    sum += outputMatrix[layer - 1][prevNeuron] * weightMatrix[layer][neuron][prevNeuron];
		}

		// zastosowanie funkcji aktywacji dla sumy na tym neuronie
		outputMatrix[layer][neuron] = sigmoid(sum);

		// uproszczona pochodna funkcji aktywacji
		outputDerivativeMatrix[layer][neuron] = (outputMatrix[layer][neuron]
			* (1 - outputMatrix[layer][neuron]));

		// outputMatrix[layer][neuron] = reLU(sum);
		// outputDerivativeMatrix[layer][neuron] =
		// reluDerivative(outputMatrix[layer][neuron]);

	    }
	}
	// wartosci output dla ostatniej warstwy
	return outputMatrix[TOTAL_NUMBER_OF_LAYERS - 1];
    }

    // eta takie n z laseczka - learning rate
    // stopien w jakim następuje poszukiwanie minimum - jak duża zmiana może
    // nastąpić co epoke
    public void train(double[] input, double[] target, double eta) throws Exception {
	if (input.length != INPUT_LAYER_SIZE || target.length != OUTPUT_LAYER_SIZE)
	    return;
	calculateOutputMatrix(input);
	backpropageteError(target);
	updateWeightMatrix(eta);
    }

    public void trainThroughWholeSet(TrainSet set, int loops, int batchSize) throws Exception {
	for (int i = 0; i < loops; i++) {

	    TrainSet batch = set.extractBatch(batchSize);

	    for (int b = 0; b < batchSize; b++) {
		this.train(batch.getInputFromSet(b), batch.getOutputFromSet(b), 0.05);
	    }
	}
    }

    public void backpropageteError(double[] target) {

	// iterowanie po neuronach ostatniej warstwy
	for (int neuron = 0; neuron < LAYER_SIZES[TOTAL_NUMBER_OF_LAYERS - 1]; neuron++) {
	    // sygnal błędu na danym neuronie można obliczyć:
	    // (output-desiredOutput)*outputDeriva
	    errorSignalMatrix[TOTAL_NUMBER_OF_LAYERS - 1][neuron] = (outputMatrix[TOTAL_NUMBER_OF_LAYERS - 1][neuron]
		    - target[neuron]) * outputDerivativeMatrix[TOTAL_NUMBER_OF_LAYERS - 1][neuron];
	}

	// iterowanie od konca po HIDDEN LAYERS i modyfikowanie wag za pomocą
	// errorInfo
	for (int layer = TOTAL_NUMBER_OF_LAYERS - 2; layer > 0; layer--) {
	    // iterowanie po kazdym neuronie warstwy - będąc na ostatniej HIDDEN
	    // LAYER...
	    for (int neuron = 0; neuron < LAYER_SIZES[layer]; neuron++) {
		double sum = 0;

		for (int nextNeuron = 0; nextNeuron < LAYER_SIZES[layer + 1]; nextNeuron++) {
		    sum += weightMatrix[layer + 1][nextNeuron][neuron] * errorSignalMatrix[layer + 1][nextNeuron];
		}

		this.errorSignalMatrix[layer][neuron] = sum * outputDerivativeMatrix[layer][neuron];
	    }
	}
    }

    public void updateWeightMatrix(double eta) {
	for (int layer = 1; layer < TOTAL_NUMBER_OF_LAYERS; layer++) {
	    for (int neuron = 0; neuron < LAYER_SIZES[layer]; neuron++) {
		for (int prevNeuron = 0; prevNeuron < LAYER_SIZES[layer - 1]; prevNeuron++) {
		    // weights[layer][neuron][prevNeuron]
		    double delta = -eta * outputMatrix[layer - 1][prevNeuron] * errorSignalMatrix[layer][neuron];
		    weightMatrix[layer][neuron][prevNeuron] += delta;
		}
		double delta = -eta * errorSignalMatrix[layer][neuron];
		biasMatrix[layer][neuron] += delta;
	    }
	}
    }

    // https://isaacchanghau.github.io/post/activation_functions/

    // funkcja aktywacji można sigmoid albo ReLU
    private double sigmoid(double x) {
	return 1d / (1 + Math.exp(-x));
    }

    @SuppressWarnings("unused")
    private double reLU(double x) {
	// System.out.println("RELU " + ((x > 0) ? x : 0d));
	return x > 0 ? x : 0d;
    }

    @SuppressWarnings("unused")
    private double reluDerivative(double x) {
	return x > 0 ? 1d : 0d;
    }

    public static void main(String[] args) throws Exception {
    	


	double[] smile1 = loadImageAsDoubleArray(new File("img/smile1.bmp"));
	double[] smile2 = loadImageAsDoubleArray(new File("img/smile2.bmp"));
	double[] smile3 = loadImageAsDoubleArray(new File("img/smile3.bmp"));
	double[] smile4 = loadImageAsDoubleArray(new File("img/smile4.bmp"));
	double[] smile5 = loadImageAsDoubleArray(new File("img/smile5.bmp"));
	double[] smile6 = loadImageAsDoubleArray(new File("img/smile6.bmp"));

	double[] sad1 = loadImageAsDoubleArray(new File("img/sad1.bmp"));
	double[] sad2 = loadImageAsDoubleArray(new File("img/sad2.bmp"));
	double[] sad3 = loadImageAsDoubleArray(new File("img/sad3.bmp"));
	double[] sad4 = loadImageAsDoubleArray(new File("img/sad4.bmp"));
	double[] sad5 = loadImageAsDoubleArray(new File("img/sad5.bmp"));
	double[] sad6 = loadImageAsDoubleArray(new File("img/sad6.bmp"));

	double[] notFace1 = loadImageAsDoubleArray(new File("img/notFace1.bmp"));
	double[] notFace2 = loadImageAsDoubleArray(new File("img/notFace2.bmp"));
	double[] notFace3 = loadImageAsDoubleArray(new File("img/notFace3.bmp"));
	double[] notFace4 = loadImageAsDoubleArray(new File("img/notFace4.bmp"));
	double[] notFace5 = loadImageAsDoubleArray(new File("img/notFace5.bmp"));
	double[] notFace6 = loadImageAsDoubleArray(new File("img/notFace6.bmp"));

	//Network net = new Network(smile1.length, 49, 49, 7, 2);
	int[] layers = new int[args.length];
	
	System.out.println(smile1.length);
	for(int i=0; i<args.length;i++) {
	 layers[i] = Integer.parseInt(args[i]);
	 System.out.println(layers[i]);
	}
	
	Network net = new Network(layers);

	TrainSet set = new TrainSet(smile1.length, 2);

	// zestaw do nauki XOR
	// Network net = new Network(2, 3, 3, 1);
	// TrainSet set = new TrainSet(2, 1);
	// set.addData(new double[] { 0.0, 0.0 }, new double[] { 0.0 });
	// set.addData(new double[] { 0.0, 1.0 }, new double[] { 1.0 });
	// set.addData(new double[] { 1.0, 0.0 }, new double[] { 1.0 });
	// set.addData(new double[] { 1.0, 1.0 }, new double[] { 0.0 });

	set.addData(smile1, new double[] { 1.0, 0.0 });
	set.addData(smile2, new double[] { 1.0, 0.0 });
	set.addData(smile3, new double[] { 1.0, 0.0 });
	set.addData(smile4, new double[] { 1.0, 0.0 });
	set.addData(smile5, new double[] { 1.0, 0.0 });
	set.addData(smile6, new double[] { 1.0, 0.0 });

	set.addData(sad1, new double[] { 0.0, 1.0 });
	set.addData(sad2, new double[] { 0.0, 1.0 });
	set.addData(sad3, new double[] { 0.0, 1.0 });
	set.addData(sad4, new double[] { 0.0, 1.0 });
	set.addData(sad5, new double[] { 0.0, 1.0 });
	set.addData(sad6, new double[] { 0.0, 1.0 });

	set.addData(notFace1, new double[] { 0.0, 0.0 });
	set.addData(notFace2, new double[] { 0.0, 0.0 });
	set.addData(notFace3, new double[] { 0.0, 0.0 });
	set.addData(notFace4, new double[] { 0.0, 0.0 });
	set.addData(notFace5, new double[] { 0.0, 0.0 });
	set.addData(notFace6, new double[] { 0.0, 0.0 });

	net.trainThroughWholeSet(set, 200000, 18);

	for (int i = 0; i < set.size(); i++) {
	    String name = "";
	    if (i < 6)
		name = "Smile";
	    if (i >= 6 && i < 12)
		name = "Sad";
	    if (i >= 12 && i < 18)
		name = "NotFace";

	    System.out.println(String.format("%-25s -> %-40s", name + i,
		    Arrays.toString(net.calculateOutputMatrix(set.getInputFromSet(i)))));
	}

	double[] testSmile = loadImageAsDoubleArray(new File("img/testSmile.bmp"));
	double[] testSmile2 = loadImageAsDoubleArray(new File("img/testSmile2.bmp"));

	double[] testSad1 = loadImageAsDoubleArray(new File("img/testSad1.bmp"));
	double[] testSad2 = loadImageAsDoubleArray(new File("img/testSad2.bmp"));

	System.out.println("TestSmile1 -> " + Arrays.toString(net.calculateOutputMatrix(testSmile)));
	System.out.println("TestSmile2 -> " + Arrays.toString(net.calculateOutputMatrix(testSmile2)));
	System.out.println("TestSad1 -> " + Arrays.toString(net.calculateOutputMatrix(testSad1)));
	System.out.println("TestSad2 -> " + Arrays.toString(net.calculateOutputMatrix(testSad2)));

    }

    public static double[] loadImageAsDoubleArray(File file) throws IOException {
	BufferedImage bImage = ImageIO.read(file);
	ByteArrayOutputStream bos = new ByteArrayOutputStream();
	ImageIO.write(bImage, "bmp", bos);
	byte[] data = bos.toByteArray();
	bos.reset();

	// System.out.println(Arrays.toString(data));

	double[] dataAsDouble = new double[data.length];

	for (int i = 0; i < data.length; i++) {
	    double d = data[i];
	    dataAsDouble[i] = d;
	}

	// System.out.println(Arrays.toString(dataAsDouble));
	return dataAsDouble;
    }

}
