package nl.uva;

import org.apache.hadoop.io.IntWritable;

/**
 * This class is for allowing mappers to emit their data in ascending order The
 * only method to override is the compareTo
 *
 * @author S. Koulouzis
 */
public class ScoreWritable extends IntWritable {

    public ScoreWritable(int score) {
        super(score);
    }

    public ScoreWritable() {
    }

    @Override
    public int compareTo(Object o) {
        int thisValue = get();
        int thatValue = ((IntWritable) o).get();
        return (thisValue > thatValue ? -1 : (thisValue == thatValue ? 0 : 1));
    }
}
