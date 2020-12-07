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

def calculate_codebook_dl(codebook):
    """
    Given a codebook, calculate its description length:
    Length(encoding) + Size(Huffman Tree)

    The bit length of the encoding is easily calculated, but the 
    size of the Huffman Tree can be represented simply as 
    the number of bits required to recover the tree. In canonical form,
    it's just the number of bits needed to encode the bit LENGTHS of
    each symbol.
    """
    codec = HuffmanCodec.from_frequencies(codebook)
    codec.encode(codebook)
    #codec.print_code_table()
    
    # Need to update this to get code length of trajectories
    entire_code = []
    for skill, occurences in codebook.items():
        entire_code.extend([skill] * occurences)
    encoded = codec.encode(entire_code)
    entire_code = set(entire_code)
    
    number_of_bits = 0
    # Calculate the number of bits to encode the symbols in the trajectory
    for symbol, (bits, val) in codec._table.items():
        if symbol in entire_code:
            number_of_bits += len(symbol.encode('utf-8')) * 8
            number_of_bits += bits
    return len(encoded)*8 + number_of_bits # * 8 for byte to bit conversion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate MDL of a directory of codebooks which are encoded in .npy format')
    parser.add_argument('location', type=str,
                        help='a file location where codebooks are stored')
    args = parser.parse_args()

    codebooks = discover_codebooks(args.location)
    for codebook in codebooks:
        dl = calculate_codebook_dl(codebook[1])
        print(codebook[0], dl)
