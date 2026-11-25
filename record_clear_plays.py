"""Record successful evaluation episodes using the frozen study source/model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--drive-out')
    ap.add_argument('--fps', type=int, default=10)
    args = ap.parse_args()
    assert args.fps > 0
    study, work, out = map(Path, (args.study, args.work, args.out))
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, expected in json.loads((study / 'source_hashes.json').read_text()).items():
        src = study / 'source' / name
        assert digest(src) == expected, 'Frozen source mismatch: ' + name
        dest = work / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    sys.path.insert(0, str(work.resolve()))
    import numpy as np
    import torch
    import pygame
    import imageio.v2 as imageio
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    from record_plays import load, _font

    torch.set_num_threads(1)
    font, small = _font(24), _font(16)
    font_path = next((p for p in (
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        'C:/Windows/Fonts/meiryob.ttc') if Path(p).exists()), None)
    results = json.loads((study / 'results.json').read_text())
    cases = [dict(result=r, index=i, eval_seed=r['eval']['seed0'] + i,
                  score=r['eval']['scores'][i], steps=r['eval']['steps'][i])
             for r in results for i, cleared in enumerate(r['eval']['clears']) if cleared]
    assert cases, 'No successful evaluation episodes are recorded'
    cases.sort(key=lambda c: (-c['score'], c['steps'], c['result']['seed'], c['eval_seed']))
    records = []

    def frame(env, result, case, final=False):
        env._update_planned_path()  # Update only the navigation drawing for RGB output.
        im = Image.fromarray(env.render())
        canvas = Image.new('RGB', (im.width, im.height + 86), (18, 21, 40))
        canvas.paste(im, (0, 86))
        draw = ImageDraw.Draw(canvas)
        title = 'CLEAR! エサをすべて回収' if final else 'ゴースト4体・改善版｜クリアプレイ'
        draw.text((12, 5), title, font=font, fill=(255, 225, 100))
        draw.text((12, 41), f"学習seed {result['seed']} / 評価seed {case['eval_seed']} / 最終 {case['score']:,}点",
                  font=small, fill=(226, 232, 242))
        draw.text((12, 62), f"Step {env.steps} / 残機 {env.lives} / エサ {env.dots_eaten}/{env.total_dots} / 貪欲方策",
                  font=small, fill=(170, 202, 226))
        return np.asarray(canvas)

    for case in cases:
        result = case['result']
        run = study / 'runs' / result['id']
        receipt = json.loads((run / 'saved_result.json').read_text())
        assert digest(run / 'model.pt') == receipt['sha256']['model.pt']
        assert digest(run / 'result.json') == receipt['sha256']['result.json']
        saved, env, agent = load(str(run))
        assert saved == result
        obs, _ = env.reset(seed=case['eval_seed'])
        env.render()  # Initialise fonts; replacing them changes presentation only.
        if font_path:
            env.font = pygame.font.Font(font_path, 14)
            env.font_big = pygame.font.Font(font_path, 18)
        name = f"clear_ghost4_train{result['seed']}_eval{case['eval_seed']}.mp4"
        path = out / name
        frames = 0
        with imageio.get_writer(str(path), fps=args.fps, codec='libx264',
                               pixelformat='yuv420p', macro_block_size=2,
                               ffmpeg_params=['-crf', '20', '-movflags', '+faststart']) as writer:
            while True:
                env.set_qvalues(agent.get_qvalues(obs))
                writer.append_data(frame(env, result, case))
                frames += 1
                obs, _, terminated, truncated, info = env.step(agent.select_action(obs, training=False))
                if terminated or truncated:
                    break
            assert info['cleared'] and env.steps == case['steps'] and info['score'] == case['score'], {
                'expected': (case['score'], case['steps']), 'actual': (info['score'], env.steps, info['cleared'])}
            last = frame(env, result, case, final=True)
            for _ in range(3 * args.fps):
                writer.append_data(last)
                frames += 1
        # Decode the full file, not just its header, to catch truncated/corrupt video.
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error', '-i', str(path),
                        '-f', 'null', '-'], check=True)
        reader = imageio.get_reader(str(path))
        assert reader.count_frames() == frames
        metadata = reader.get_meta_data()
        reader.close()
        record = dict(file=name, model_id=result['id'], model_sha256=receipt['sha256']['model.pt'],
                      train_seed=result['seed'], eval_seed=case['eval_seed'], score=info['score'],
                      cleared=True, steps=env.steps, lives_left=env.lives,
                      fps=args.fps, frames=frames, duration_seconds=frames / args.fps,
                      size=list(metadata['size']), sha256=digest(path), reproduced_evaluation=True)
        records.append(record)
        if len(records) == 1:
            shutil.copy2(path, out / 'best_clear_ghost4.mp4')
            Image.fromarray(last).save(out / 'best_clear_ghost4.png')
        env.close()
        print('VIDEO_VERIFIED', name, info['score'], env.steps, frames, flush=True)
    (out / 'clear_videos.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'README.md').write_text(
        '# ゴースト4体のクリア動画\n\n'
        '学習済みモデルと評価seedを用いて、保存済みのクリア7回を再現しました。'
        '得点・ステップ数・クリア結果が元の評価と一致することを検証済みです。\n\n'
        'best_clear_ghost4.mp4 は2,920点・329ステップのプレイです。10fps、最後に3秒静止。'
        '動画は新しい学習ではなく、ランダムな行動を使わない評価の再現です。\n', encoding='utf-8')
    if args.drive_out:
        drive_out = Path(args.drive_out)
        drive_out.mkdir(parents=True, exist_ok=True)
        for p in out.iterdir():
            if p.is_file():
                dest = drive_out / p.name
                assert not dest.exists(), 'Preserve existing video: ' + str(dest)
                shutil.copy2(p, dest)
                assert digest(dest) == digest(p)
        print('DRIVE_VIDEOS_SYNCED', drive_out, flush=True)
    archive = out.parent / '000_ghost4_clear_videos_20261007.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as z:
        for p in sorted(out.iterdir()):
            if p.is_file():
                z.write(p, p.name)
    print('VIDEOS_COMPLETE', len(records), archive, archive.stat().st_size, digest(archive), flush=True)


if __name__ == '__main__':
    main()
