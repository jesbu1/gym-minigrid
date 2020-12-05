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
    """
    codec = HuffmanCodec.from_frequencies(codebook)
    codec.encode(codebook)
    #codec.print_code_table()

    entire_code = []
    for skill, occurences in codebook.items():
        entire_code.extend([skill] * occurences)
    #message = "".join(entire_code)
    encoded = codec.encode(entire_code)
    #print(len(encoded))
    #print(encoded)

    #currently only calculates the length of encoding the same set of skills
    return len(encoded)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate MDL of a directory of codebooks which are encoded in .npy format')
    parser.add_argument('location', type=str,
                        help='a file location where codebooks are stored')
    args = parser.parse_args()

    codebooks = discover_codebooks(args.location)
    for codebook in codebooks:
        dl = calculate_codebook_dl(codebook[1])
        print(codebook[0], dl)