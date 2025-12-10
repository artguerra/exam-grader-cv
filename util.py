def xor_encrypt_to_hex(plaintext: str, key: str) -> str:
    data = plaintext.encode("utf-8")
    key_bytes = key.encode("utf-8")
    out = bytearray(len(data))

    for i, b in enumerate(data):
        out[i] = b ^ key_bytes[i % len(key_bytes)]
    return out.hex()


def xor_decrypt_from_hex(cipher_hex: str, key: str) -> str:
    data = bytes.fromhex(cipher_hex)
    key_bytes = key.encode("utf-8")
    out = bytearray(len(data))

    for i, b in enumerate(data):
        out[i] = b ^ key_bytes[i % len(key_bytes)]
    return out.decode("utf-8")


def mm_to_px(size_mm: float, dpi: int) -> int:
    return int(size_mm * (dpi / 72.0))
