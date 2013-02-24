package nl.uva;

import java.util.Comparator;
import org.biojava3.core.sequence.ProteinSequence;

class ValueComparator implements Comparator<ProteinSequence> {

    java.util.Map<ProteinSequence, Integer> base;

    public ValueComparator(java.util.Map<ProteinSequence, Integer> base) {
        this.base = base;
    }

    @Override
    public int compare(ProteinSequence a, ProteinSequence b) {
        if (base.get(a) >= base.get(b)) {
            return -1;
        } else {
            return 1;
        } // returning 0 would merge keys
    }
}
