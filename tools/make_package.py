# 打包脚本：把 SSTV Studio 的全部相关文件集中到一个文件夹里。
#
# 用法：  python -X utf8 tools\make_package.py
#
# 只复制运行所需要的文件，并剔除临时文件、各种缓存和与本项目无关的文件。

from __future__ import annotations

import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sstv import version as version_module  # noqa: E402

# 版本号只有一个来源（sstv/version.py），文件夹名和 zip 名都由它推导出来，
# 所以不会出现“包叫 1.1.0、程序里写 1.0.0”这种对不上的情况。
PKG_NAME = f"SSTV-Studio-{version_module.VERSION}"
PKG = os.path.join(ROOT, PKG_NAME)

# 顶层文件
FILES = [
    "run.bat",
    "run-zh.bat",
    "sstv_app.py",
    "sstv_bootstrap.py",
    "README.md",
    "README-zh.md",
]

# 需要整目录复制的内容
DIRS = [
    "sstv",
    "vendor",
    "tools",
]

# 验证脚本：只带正式的校验脚本，不带调试过程中用过的临时脚本
TEST_DIR = "tests"
TEST_KEEP_PREFIX = "check_"

# 示例图片
SAMPLES = [
    ("out/final-src-Robot36.png", "out/示例-原图-Robot36.png"),
    ("out/final-dec-Robot36.png", "out/示例-解码-Robot36.png"),
    ("out/final-src-PD120.png", "out/示例-原图-PD120.png"),
    ("out/final-dec-PD120.png", "out/示例-解码-PD120.png"),
    ("out/grad-Robot36.png", "out/往返测试-Robot36.png"),
    ("out/grad-PD120.png", "out/往返测试-PD120.png"),
    ("out/grad-ScottieDX.png", "out/往返测试-ScottieDX.png"),
    ("out/瀑布图-界面.png", "out/示例-瀑布图.png"),
    ("out/瀑布图-带刻度.png", "out/示例-瀑布图-带刻度.png"),
]

# 复制时一律跳过的目录名
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache"}
# 复制时一律跳过的文件名后缀
SKIP_SUFFIX = (".pyc", ".pyo")


def clean(path: str) -> None:
    """删除目录，遇到权限问题（例如沙箱挡住的临时目录）就跳过。"""
    if not os.path.exists(path):
        return
    def on_error(func, target, exc_info):
        print(f"    跳过无法删除的 {target}")
    shutil.rmtree(path, onerror=on_error)


def ignore(directory, names):
    skipped = []
    for name in names:
        if name in SKIP_DIRS or name.endswith(SKIP_SUFFIX):
            skipped.append(name)
    return skipped


def copy_tree(src: str, dst: str) -> int:
    count = 0
    for base, _dirs, files in os.walk(src):
        for name in files:
            if name.endswith(SKIP_SUFFIX):
                continue
            source = os.path.join(base, name)
            relative = os.path.relpath(source, src)
            if any(part in SKIP_DIRS for part in relative.split(os.sep)):
                continue
            target = os.path.join(dst, relative)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(source, target)
            count += 1
    return count


def strip_caches(root: str) -> int:
    """Delete every cache directory left under *root*.

    Running the program inside the package (which the verification step does)
    creates ``__pycache__`` folders.  They must not go into the archive: they are
    build artefacts, they bloat it, and their contents are meaningless on another
    machine.
    """
    removed = 0
    for base, dirs, _files in os.walk(root, topdown=False):
        for name in list(dirs):
            if name in SKIP_DIRS:
                target = os.path.join(base, name)
                clean(target)
                removed += 1
    return removed


def main() -> int:
    print("=" * 68)
    print("打包 SSTV Studio")
    print("=" * 68)

    print(f"\n目标文件夹: {PKG_NAME}")
    if os.path.exists(PKG):
        print("  已存在，先清空…")
        clean(PKG)
    os.makedirs(PKG, exist_ok=True)

    total = 0

    print("\n1. 复制程序文件")
    for name in FILES:
        src = os.path.join(ROOT, name)
        if not os.path.exists(src):
            print(f"  缺少 {name}，跳过")
            continue
        shutil.copy2(src, os.path.join(PKG, name))
        print(f"  + {name}")
        total += 1

    print("\n2. 复制程序包和依赖库")
    for name in DIRS:
        src = os.path.join(ROOT, name)
        if not os.path.isdir(src):
            print(f"  缺少 {name}/，跳过")
            continue
        count = copy_tree(src, os.path.join(PKG, name))
        print(f"  + {name}/  ({count} 个文件)")
        total += count

    print("\n3. 复制验证脚本")
    src_tests = os.path.join(ROOT, TEST_DIR)
    dst_tests = os.path.join(PKG, TEST_DIR)
    os.makedirs(dst_tests, exist_ok=True)
    kept = 0
    for name in sorted(os.listdir(src_tests)):
        if not name.startswith(TEST_KEEP_PREFIX) or not name.endswith(".py"):
            continue
        shutil.copy2(os.path.join(src_tests, name), os.path.join(dst_tests, name))
        kept += 1
    print(f"  + {TEST_DIR}/  ({kept} 个校验脚本)")
    total += kept

    print("\n4. 复制示意图片")
    dst_out = os.path.join(PKG, "out")
    os.makedirs(dst_out, exist_ok=True)
    for source, target in SAMPLES:
        src = os.path.join(ROOT, source)
        if not os.path.exists(src):
            print(f"  缺少 {source}，跳过")
            continue
        shutil.copy2(src, os.path.join(PKG, target))
        print(f"  + {target}")
        total += 1

    print("\n5. 清理缓存目录")
    stripped = strip_caches(PKG)
    print(f"  删除 {stripped} 个缓存目录（__pycache__）")

    print("\n6. 统计")
    size = 0
    files = 0
    for base, _dirs, names in os.walk(PKG):
        for name in names:
            files += 1
            try:
                size += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    print(f"  共 {files} 个文件，{size / 1024 / 1024:.1f} MB")
    print(f"  位置: {PKG}")

    print("\n7. 校验清单")
    required = [
        "run.bat", "run-zh.bat", "sstv_app.py", "sstv_bootstrap.py",
        "README.md", "README-zh.md",
        "sstv/modes.py", "sstv/encoder.py", "sstv/decoder.py", "sstv/dsp.py",
        "sstv/audio.py", "sstv/gui.py", "sstv/vis.py", "sstv/colorspace.py",
        "sstv/waterfall.py", "sstv/wasapi.py", "sstv/streaming.py",
        "sstv/version.py",
        "tools/make_package.py", "tools/bump_version.py",
        "vendor/numpy/__init__.py", "vendor/PIL/Image.py",
        "tests/check_acceptance.py", "tests/check_waterfall.py",
        "tests/check_loopback.py", "tests/check_streaming.py",
        "tests/check_inferred_mode.py",
    ]
    missing = [r for r in required if not os.path.exists(os.path.join(PKG, r.replace("/", os.sep)))]
    if missing:
        print("  缺少以下必需文件：")
        for name in missing:
            print(f"    - {name}")
        return 1
    print(f"  运行所需的 {len(required)} 项文件全部就位")

    print("\n8. 打包成 zip")
    zip_path = make_zip(PKG)
    if zip_path:
        print(f"  + {os.path.basename(zip_path)}  "
              f"({os.path.getsize(zip_path) / 1024 / 1024:.1f} MB)")

    print("\n打包完成。")
    return 0


def make_zip(pkg: str) -> str | None:
    """把文件夹压缩成一个 zip，条目名使用标准正斜杠。

    不用 PowerShell 的 Compress-Archive：它在 Windows 上写出的是反斜杠路径，
    那不是 zip 规范要求的写法，跨平台解压时容易出问题。
    """
    import zipfile

    zip_path = pkg + ".zip"
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except OSError as exc:
            print(f"  无法删除旧压缩包: {exc}")
            return None
    parent = os.path.dirname(pkg)
    name = os.path.basename(pkg)
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for base, dirs, files in os.walk(pkg):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for filename in sorted(files):
                    if filename.endswith(SKIP_SUFFIX):
                        continue
                    full = os.path.join(base, filename)
                    arcname = os.path.join(name, os.path.relpath(full, pkg))
                    archive.write(full, arcname.replace(os.sep, "/"))
    except OSError as exc:
        print(f"  压缩失败: {exc}")
        return None
    return zip_path


if __name__ == "__main__":
    raise SystemExit(main())
