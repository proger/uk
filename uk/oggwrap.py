from pathlib import Path
import argparse, struct, random, sys
from typing import List


def ogg_crc(buf: bytes) -> int:
    """
    Compute the Ogg CRC32 (non-reflected, major-order CRC) over the data.
    This matches libogg's crc implementation (polynomial 0x04C11DB7).
    """
    reg = 0
    for b in buf:
        reg ^= (b << 24) & 0xFFFFFFFF
        for _ in range(8):
            if reg & 0x80000000:
                reg = ((reg << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                reg = (reg << 1) & 0xFFFFFFFF
    return reg


def make_page(packets: List[bytes],
              *,
              continued: bool = False,
              bos: bool = False,
              eos: bool = False,
              granule: int,
              serial: int,
              seq: int) -> bytes:
    """Build one Ogg page from 1 ≤ n ≤ 255 laced packets."""
    hdr_type = 0
    if continued: hdr_type |= 0x01
    if bos:        hdr_type |= 0x02
    if eos:        hdr_type |= 0x04

    seg_sizes, seg_data = [], b''
    for pkt in packets:
        for i in range(0, len(pkt), 255):
            seg = pkt[i : i + 255]
            seg_sizes.append(len(seg))
            seg_data += seg

    if len(seg_sizes) > 255:
        raise ValueError("page would exceed 255 lacing values")

    header = bytearray(b'OggS')          # capture pattern
    header.extend(b'\x00')               # stream structure version
    header.extend(bytes([hdr_type]))
    header.extend(struct.pack('<Q', granule))
    header.extend(struct.pack('<I', serial))
    header.extend(struct.pack('<I', seq))
    header.extend(b'\x00\x00\x00\x00')   # CRC placeholder
    header.extend(bytes([len(seg_sizes)]))
    header.extend(bytes(seg_sizes))

    page = bytes(header) + seg_data
    # CRC calculation: field MUST be zeroed first (already is)
    crc = ogg_crc(page)
    page = page[:22] + struct.pack('<I', crc) + page[26:]
    return page


def build_opus_head(channels: int, pre_skip: int, sample_rate: int,
                    gain_q8: int = 0, mapping_family: int = 0) -> bytes:
    return (
        b'OpusHead' +
        bytes([1, channels]) +
        struct.pack('<H', pre_skip) +
        struct.pack('<I', sample_rate) +
        struct.pack('<h', gain_q8) +
        bytes([mapping_family])
    )


def build_opus_tags(vendor: bytes = b'') -> bytes:
    return (
        b'OpusTags' +
        struct.pack('<I', len(vendor)) + vendor +   # vendor string
        struct.pack('<I', 0)                        # 0 user comments
    )


def write_ogg_opus(packets: List[bytes], output: Path,
                   sample_rate: int = 48_000, frame_dur_s: float = 0.06):
    sample_rate        = 48_000
    frame_len_samples  = int(0.06 * sample_rate)          # 60 ms → 2 880
    serial             = random.randrange(1, 2**32)       # 0 is legal but unusual
    seqno              = 0                                # page sequence counter

    opus_head = (
        b'OpusHead' +
        bytes([1, 1]) +                # version=1, channels=1
        struct.pack('<H', 0) +         # pre-skip
        struct.pack('<I', sample_rate) +
        struct.pack('<h', 0) +         # output-gain (q8)
        b'\x00'                        # mapping-family
    )

    opus_tags = (
        b'OpusTags' +
        struct.pack('<I', 0) +         # 0-byte vendor string
        struct.pack('<I', 0)           # 0 user comments
    )

    with open(output, 'wb') as f:
        # 1) identification page – BOS only
        f.write(make_page([opus_head],
                          bos=True,
                          granule=0,
                          serial=serial,
                          seq=seqno));      seqno += 1

        # 2) comment page – no flags
        f.write(make_page([opus_tags],
                          granule=0,
                          serial=serial,
                          seq=seqno));      seqno += 1

        # 3 +) audio pages
        for idx, pkt in enumerate(packets):
            granule = (idx + 1) * frame_len_samples
            f.write(make_page([pkt],
                              eos = (idx == len(packets) - 1),
                              granule = granule,
                              serial = serial,
                              seq = seqno))
            seqno += 1


def read_packets(bin_file, offset, duration) -> list[bytes]:
    # frame size is fixed under CBR: 16 kbps @ 60 ms ⇒ 16e3*0.06/8 = 120 bytes
    frame_bytes = int(0.06 * 16000 / 8)
    assert frame_bytes == 120
    start = offset * frame_bytes
    total_bytes = duration * frame_bytes
    packets = []
    with open(bin_file, 'rb') as fd:
        fd.seek(start)
        data = fd.read(total_bytes)
        if len(data) != total_bytes:
            raise ValueError(f'Unexpected data length: got {len(data)}, expected {total_bytes}\n')
        for i in range(0, len(data), frame_bytes):
            packets.append(data[i:i+frame_bytes])
    return packets


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('bin_file',  type=Path,
                    help='raw Opus CELT packet file (from prepare_opusbin)')
    ap.add_argument('offset',    type=int,
                    help='frame offset (60 ms units)')
    ap.add_argument('duration',  type=int,
                    help='number of 60 ms frames to include')
    ap.add_argument('output',    type=Path, help='output .ogg path')
    args = ap.parse_args()

    packets = read_packets(args.bin_file, args.offset, args.duration)
    write_ogg_opus(packets, args.output)
    from uk.subprocess import sh
    sh('ffprobe', '-i', args.output)
