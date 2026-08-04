# unicode_map.py
def bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping of UTF-8 byte integers to displayable Unicode strings,
    following the standard GPT-2 byte-level representation.

    Returns:
        Dictionary mapping byte integers (0-255) to Unicode string characters.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs_str = [chr(x) for x in cs]
    return dict(zip(bs, cs_str))
