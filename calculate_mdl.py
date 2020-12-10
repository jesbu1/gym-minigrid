from dahuffman import HuffmanCodec
import numpy as np
import argparse
import os

def discover_codebooks(location):
    """
    Return list of codebooks given the directory location
    """
    codebooks = []
    for codebook_file in os.listdir(location):
        if codebook_file.endswith('.npy'):
            with open(os.path.join(location, codebook_file), 'rb+') as f:
                codebook = np.load(f, allow_pickle=True)
            codebooks.append((codebook_file, codebook.item())) #.item() to extract dictionary from 0d array
    return codebooks

def preprocess_codebook(codebook):
    """
    Removes trajectories from keys and puts the trajectories into single
    symbol list format. Also removes skills with 0 frequency.
    """
    trajectories = codebook.pop('trajectories')
    codebook_with_spaces = {}
    for key, value in codebook.items():
        if key == 'probabilities' or key == 'length_range':
            continue  # skip these
        if value > 0:
            codebook_with_spaces[key] = value

    single_symbol_traj_split = []
    for trajectory in trajectories:
        for symbol in trajectory.split(" "):
            if symbol != "":
                single_symbol_traj_split.append(symbol)
    #trajectories = "".join(trajectories)
    return single_symbol_traj_split, codebook_with_spaces
    

def calculate_codebook_dl(codebook):
    """
    Given a codebook, calculate its description length: Length(encoding) +
    Size(Huffman Tree)

    
    The bit length of the encoding is easily calculated, but the size of the
    Huffman Tree can be represented simply as the number of bits required to
    recover the tree. In canonical form, it's just the number of bits needed
    to encode the bit LENGTHS of each symbol.
    """
    trajectories, codebook = preprocess_codebook(codebook)
    codec = HuffmanCodec.from_frequencies(codebook)
    codec.encode(codebook)
    #codec.print_code_table()
    encoded = codec.encode(trajectories)

    trajectory_symbol_set = set(trajectories) 
    tree_bits = 0
    # Calculate the number of bits to send the tree 
    for symbol, (bits, val) in codec._table.items():
        if symbol in trajectory_symbol_set:
            tree_bits += len(symbol.encode('utf-8')) * 8
            tree_bits += bits
    dl = len(encoded)*8 + tree_bits # * 8 for byte to bit conversion
    uncompressed_len = len("".join(trajectories).encode('utf-8')) * 8
    return dl, tree_bits, codec, uncompressed_len 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate MDL of a directory of codebooks which are encoded in .npy format')
    parser.add_argument('location', type=str,
                        help='a file location where codebooks are stored')
    args = parser.parse_args()

    codebooks = discover_codebooks(args.location)
    codebook_name_dl_tuples = []
    for codebook in codebooks:
        dl, tree_bits, codec, uncompressed_len = calculate_codebook_dl(codebook[1])
        codebook_name_dl_tuples.append((codebook[0], dl, tree_bits, codec, uncompressed_len))
    sorted_codebooks_by_dl = sorted(codebook_name_dl_tuples, key=lambda x: x[1])
    for name, dl, tree_bits, codec, uncompressed_len in sorted_codebooks_by_dl:
        #print(name, dl, uncompressed_len)
        print(name, dl)
