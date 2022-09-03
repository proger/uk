from pathlib import Path
from typing import Dict
from collections import defaultdict

from praatio import textgrid as tgio
from praatio.data_classes.interval_tier import Interval


def ctm_to_textgrid(file,
                    output_dir: Path,
                    symtab: Dict[str, str]):
    intervals = defaultdict(list)
    tg = tgio.Textgrid()

    for line in file:
        utt_id, chan, start_, duration_, label = line.split()
        start = round(float(start_), 4)
        end = round(start + float(duration_), 4)
        interval = Interval(start, end, symtab.get(label, label))

        intervals[utt_id].append(interval)

    for utt_id in intervals:
        tg = tgio.Textgrid()
        tg.minTimestamp = 0
        tg.maxTimestamp = intervals[utt_id][-1].end

        tier_name = 'phones'
        tg.addTier(tgio.IntervalTier(tier_name, [], minT=0, maxT=tg.maxTimestamp))

        for interval in intervals[utt_id]:
            tg.tierDict[tier_name].entryList.append(interval)

        tg.save(output_dir / f'{utt_id}.TextGrid',
                includeBlankSpaces=True,
                format='long_textgrid',
                reportingMode='error')
