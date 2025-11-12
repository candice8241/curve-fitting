#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速入门脚本 - 晶系判断和相变识别

这个脚本演示如何使用phase_transition_analysis模块

用法:
    python quick_start.py
"""

from phase_transition_analysis import analyze_phase_transition
import os

def main():
    print("="*70)
    print("晶系判断和相变识别 - 快速入门示例")
    print("="*70)

    # 检查示例文件是否存在
    example_file = "example_peaks.csv"

    if not os.path.exists(example_file):
        print(f"\n错误：找不到示例文件 {example_file}")
        print("请确保example_peaks.csv文件在当前目录下")
        return

    print(f"\n使用示例文件: {example_file}")
    print("\n开始分析...\n")

    try:
        # 执行分析
        results = analyze_phase_transition(example_file)

        # 显示简要结果
        print("\n" + "="*70)
        print("快速结果摘要")
        print("="*70)

        if results['transition_pressure']:
            print(f"\n✓ 检测到相变")
            print(f"  相变压力: {results['transition_pressure']:.2f} GPa")
        else:
            print("\n✗ 未检测到相变")

        if results['before_system']:
            print(f"\n相变前:")
            print(f"  晶系: {results['before_system']}")
            print(f"  晶胞参数: {results['before_params']}")

        if results['after_analysis']:
            print(f"\n相变后分析结果数量: {len(results['after_analysis'])}")

            # 显示第一个和最后一个压力点的结果
            first = results['after_analysis'][0]
            last = results['after_analysis'][-1]

            print(f"\n第一个压力点 ({first['pressure']:.2f} GPa):")
            print(f"  新峰数量: {len(first['new_peaks'])}")
            if first['new_phase_system']:
                print(f"  新相晶系: {first['new_phase_system']}")
                print(f"  晶胞参数: {first['new_phase_params']}")

            if len(results['after_analysis']) > 1:
                print(f"\n最后一个压力点 ({last['pressure']:.2f} GPa):")
                print(f"  新峰数量: {len(last['new_peaks'])}")
                if last['new_phase_system']:
                    print(f"  新相晶系: {last['new_phase_system']}")
                    print(f"  晶胞参数: {last['new_phase_params']}")

        print("\n" + "="*70)
        print("分析完成！详细结果已保存到JSON文件")
        print("="*70)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
