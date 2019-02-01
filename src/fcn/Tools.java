package fcn;

public class Tools {

    public static double[] createArray(int size, double init_value) {
	if (size < 1) {
	    return null;
	}
	double[] ar = new double[size];
	for (int i = 0; i < size; i++) {
	    ar[i] = init_value;
	}
	return ar;
    }

    public static double[] createRandomArray(int size, double lower_bound, double upper_bound) {
	if (size < 1) {
	    return null;
	}
	double[] ar = new double[size];
	for (int i = 0; i < size; i++) {
	    ar[i] = randomValue(lower_bound, upper_bound);
	}
	return ar;
    }

    public static double[][] generateRandomArray(int sizeX, int sizeY, double min, double max) throws Exception {
	if (sizeX < 1 || sizeY < 1) {
	    throw new Exception("Tablica nie moze miec wymiarow mniejszych niez 1x1");
	}
	double[][] ar = new double[sizeX][sizeY];
	for (int i = 0; i < sizeX; i++) {
	    ar[i] = createRandomArray(sizeY, min, max);
	}
	return ar;
    }

    public static double randomValue(double min, double max) {
	return Math.random() * (max - min) + min;
    }

    public static Integer[] randomValues(int lowerBound, int upperBound, int amount) {

	lowerBound--;

	if (amount > (upperBound - lowerBound)) {
	    return null;
	}

	Integer[] values = new Integer[amount];

	for (int i = 0; i < amount; i++) {
	    int n = (int) (Math.random() * (upperBound - lowerBound + 1) + lowerBound);

	    while (containsValue(values, n)) {
		n = (int) (Math.random() * (upperBound - lowerBound + 1) + lowerBound);
	    }
	    values[i] = n;
	}
	return values;
    }

    public static <T extends Comparable<T>> boolean containsValue(T[] ar, T value) {
	for (int i = 0; i < ar.length; i++) {
	    if (ar[i] != null) {
		if (value.compareTo(ar[i]) == 0) {
		    return true;
		}
	    }

	}
	return false;
    }

}
