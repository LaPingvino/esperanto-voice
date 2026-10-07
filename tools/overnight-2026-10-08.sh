#!/bin/bash
# Overnight pipeline, 2026-10-08. Every step logs to overnight.log; a failed
# step is logged and the pipeline continues with whatever still makes sense.
set -u
cd /home/joop/esperanto-kurso-gae/tts-work
V=../esperanto-voice
DS=$V/dataset
PY=venv/bin/python
export HF_HOME=$PWD/hf HF_HUB_OFFLINE=1
log() { echo "$(date +%H:%M) $*" | tee -a overnight.log; }

log "=== overnight pipeline start"

# 1. Wait for profiling: the profiler touches profiles.done when it exits.
#    Only speakers with >= 5 validated hours are profiled (48 of them).
while [ ! -e $DS/profiles.done ]; do sleep 30; done
log "1 profiling: $(($(wc -l < $DS/profiles.tsv)-1)) speakers profiled"

# 2. Pick voices; relax the filters if nobody passes.
$PY -I $V/tools/pick_speakers.py $DS/cv27-meta/speakers.tsv $DS/profiles.tsv \
    --male 2 --female 2 > $DS/picked.txt 2> $DS/shortlist.txt
if [ ! -s $DS/picked.txt ]; then
  log "2 strict filters picked nobody; relaxing"
  $PY -I $V/tools/pick_speakers.py $DS/cv27-meta/speakers.tsv $DS/profiles.tsv \
      --male 2 --female 2 --max-per 0.2 --max-nasal 15 --min-trill 0.4 --max-breath 0.5 \
      --min-snr 20 --min-rolloff 3 > $DS/picked.txt 2> $DS/shortlist.txt
fi
log "2 picked: $(cut -f2 $DS/picked.txt | tr '\n' ';')"
[ -s $DS/picked.txt ] || { log "nothing picked; stopping before extraction"; exit 1; }

# 3. Extract all good clips of the picked speakers in one streaming pass.
mkdir -p $DS/cv27-full && ln -sfn ../cv27-meta/validated.tsv $DS/cv27-full/validated.tsv
$PY -I $V/tools/cv_stream.py clips $DS/cv-corpus-27.0-eo.tar.gz $DS/cv27-full \
    --speakers $DS/picked.txt >> overnight.log 2>&1
got=$(find $DS/cv27-full/speakers -name '*.mp3' | wc -l)
log "3 extracted $got clips ($(du -sh $DS/cv27-full | cut -f1))"

# 4. Delete the archive only if extraction clearly worked (Joop has a local copy).
if [ "$got" -ge 1000 ]; then
  rm -f $DS/cv-corpus-27.0-eo.tar.gz && log "4 archive deleted; $(df -h / | tail -1 | awk '{print $4}') free"
else
  log "4 extraction looks incomplete ($got clips); archive KEPT"
fi

# 5. Training data for the best male and best female voice.
pick() { grep -P "\t$1 " $DS/picked.txt | head -1 | cut -f1; }
for g in male female; do
  id=$(pick $g); [ -n "$id" ] || { log "5 no $g voice picked"; continue; }
  $PY -I $V/tools/cv_prepare.py $DS/cv27-full/speakers/${id:0:16} data/cv-$g --max-clips 2500 \
      >> overnight.log 2>&1 && log "5 data/cv-$g: $(tail -1 overnight.log)"
  $PY -I $V/tools/select_coverage.py data/cv-$g --minutes 30 > data/cv-$g-cover30.csv 2>> overnight.log \
      && log "5 coverage: $(tail -1 overnight.log)"
done

# 6. Training: breath fix on the known recipe, then the two Common Voice voices.
E="$PY $V/tools/experiment.py"
BREATH="--extra --data.keep_seconds_before_silence 0.02"
$E --data data/karlo --minutes 5 --train-minutes 90 --name karlo5-breathfix $BREATH \
    >> overnight.log 2>&1; log "6 $(tail -1 overnight.log)"
for g in male female; do
  [ -s data/cv-$g-cover30.csv ] || continue
  $E --data data/cv-$g --csv data/cv-$g-cover30.csv --train-minutes 150 --name cv-$g-30 $BREATH \
      >> overnight.log 2>&1; log "6 $(tail -1 overnight.log)"
done

log "=== done"
cat runs/results.tsv | tee -a overnight.log
