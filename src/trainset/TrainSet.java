package trainset;

import java.util.ArrayList;
import fcn.Tools;

public class TrainSet {

    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    private ArrayList<double[][]> data = new ArrayList<>();

    public TrainSet(int INPUT_SIZE, int OUTPUT_SIZE) {
	this.INPUT_SIZE = INPUT_SIZE;
	this.OUTPUT_SIZE = OUTPUT_SIZE;
    }

    public void addData(double[] in, double[] expected) throws Exception {
	if (in.length != INPUT_SIZE)
	    throw new Exception("Nieprawidłowa liczba danych wejściowych");
	if (expected.length != OUTPUT_SIZE)
	    throw new Exception("Nieprawidłowa liczba danych oczekiwanyc.");
	data.add(new double[][] { in, expected });
    }

    public TrainSet extractBatch(int size) throws Exception {
	
	if (size > 0 && size <= this.size()) {
	    
	    TrainSet set = new TrainSet(INPUT_SIZE, OUTPUT_SIZE);
	    
	    Integer[] ids = Tools.randomValues(0, this.size() - 1, size);
	    
	    for (Integer i : ids) {
		set.addData(this.getInputFromSet(i), this.getOutputFromSet(i));
	    }
	    return set;
	} else
	    return this;
    }

    public int size() {
	return data.size();
    }

    public double[] getInputFromSet(int index) {
	if (index >= 0 && index < size())
	    return data.get(index)[0];
	else
	    return null;
    }

    public double[] getOutputFromSet(int index) {
	if (index >= 0 && index < size())
	    return data.get(index)[1];
	else
	    return null;
    }

    public int getINPUT_SIZE() {
	return INPUT_SIZE;
    }

    public int getOUTPUT_SIZE() {
	return OUTPUT_SIZE;
    }

    public void addData(byte[] data2, byte[] data3) {
	// TODO Auto-generated method stub
	
    }
}
