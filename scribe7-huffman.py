from collections import Counter
import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(text):
    freq_counter = Counter(text)
    priority_queue = [Node(char, freq) for char, freq in freq_counter.items()]
    heapq.heapify(priority_queue)
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)
    return priority_queue[0]

def build_huffman_tree2(freqs, chars):
    priority_queue = [Node(chars[i], freqs[i]) for i in range(len(freqs))]
    heapq.heapify(priority_queue)
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)
    return priority_queue[0]
def build_codes(node, prefix="", code_map=None):
    if code_map is None:
        code_map = {}
    if node:
        if node.char is not None:
            code_map[node.char] = prefix
        build_codes(node.left, prefix + "0", code_map)
        build_codes(node.right, prefix + "1", code_map)
    return code_map

def huffman_encode(text, code_map):
    return ''.join(code_map[char] for char in text)

def huffman_decode(encoded_text, root):
    decoded_text = []
    current_node = root
    for bit in encoded_text:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right
        if current_node.char is not None:
            decoded_text.append(current_node.char)
            current_node = root
    return ''.join(decoded_text)


# Example usage:
if __name__ == "__main__":
    text = "abcabc"
    print(f"Original Text: {text}")

    # Build the Huffman Tree and Code Map
    huffman_tree = build_huffman_tree2([0.25, 0.2, 0.15, 0.15, 0.25], ['a', 'b', 'c', 'd', 'e'])
    code_map = build_codes(huffman_tree)

    # Encode and Decode
    encoded_text = huffman_encode(text, code_map)
    print(f"Encoded Text: {encoded_text}")

    decoded_text = huffman_decode(encoded_text, huffman_tree)
    print(f"Decoded Text: {decoded_text}")

    # Display Huffman Codes
    print("\nCharacter Codes:")
    for char, code in code_map.items():
        print(f"{repr(char)}: {code}")
