import copy
import heapq
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. UNIFIED TASK + RESULT DATA STRUCTURES
# ---------------------------------------------------------

@dataclass
class UniversalTask:
    name: str

    # Modern / Enoki scheduler fields.
    duration_us: int = 0
    priority_weight: float = 1.0
    locality_hint: str = "default"
    app_group: str = "default"
    arrival_time_us: int = 0
    is_short: bool = False

    # Hard-real-time scheduler fields.
    period: int = 0
    run_time: int = 0
    time_remaining: int = 0
    current_deadline: int = 0

    # Simulation state.
    vruntime: float = 0.0
    first_run_time_us: int | None = None
    completion_time_us: int | None = None


@dataclass
class ModernSchedulerResult:
    name: str
    task_count: int
    total_work_us: int
    makespan_us: int
    avg_wait_us: float
    avg_turnaround_us: float
    short_avg_wait_us: float
    long_avg_wait_us: float
    context_switches: int
    context_overhead_us: int
    useful_processor_utilization: float
    simulated_crash: bool
    notes: str = ""


@dataclass
class RealTimeResult:
    name: str
    theoretical_utilization: float
    crashed: bool
    crash_tick: int | None
    crash_task: str | None
    busy_ticks: int
    idle_ticks: int


@dataclass
class SyntheticProcessor:
    context_switch_latency_us: int = 0
    busy_ticks: int = 0
    idle_ticks: int = 0
    context_switches: int = 0
    context_overhead_us: int = 0
    last_task_name: str | None = None

    def execute_tick(self, task: UniversalTask | None) -> None:
        if task:
            self.busy_ticks += 1
            task.time_remaining -= 1
        else:
            self.idle_ticks += 1

    def apply_dispatch_penalty(self, task_name: str | None) -> int:
        if task_name and task_name != self.last_task_name:
            self.context_switches += 1
            self.context_overhead_us += self.context_switch_latency_us
            self.last_task_name = task_name
            return self.context_switch_latency_us
        return 0


# ---------------------------------------------------------
# 2. SHARED HELPERS
# ---------------------------------------------------------

def theoretical_utilization(tasks: list[UniversalTask]) -> float:
    return sum(t.run_time / t.period for t in tasks if t.period) * 100


def finish_modern_result(
    name: str,
    tasks: list[UniversalTask],
    makespan_us: int,
    processor: SyntheticProcessor,
    processor_count: int = 1,
    notes: str = "",
) -> ModernSchedulerResult:
    total_work_us = sum(t.duration_us for t in tasks)
    waits = [
        (t.completion_time_us or 0) - t.arrival_time_us - t.duration_us
        for t in tasks
    ]
    turnarounds = [
        (t.completion_time_us or 0) - t.arrival_time_us
        for t in tasks
    ]
    short_waits = [wait for wait, task in zip(waits, tasks) if task.is_short]
    long_waits = [wait for wait, task in zip(waits, tasks) if not task.is_short]
    useful_time = total_work_us
    lost_time = processor.context_overhead_us
    useful_util = (
        (useful_time / (useful_time + lost_time)) * 100
        if useful_time + lost_time
        else 0
    )

    return ModernSchedulerResult(
        name=name,
        task_count=len(tasks),
        total_work_us=total_work_us,
        makespan_us=makespan_us,
        avg_wait_us=sum(waits) / len(waits),
        avg_turnaround_us=sum(turnarounds) / len(turnarounds),
        short_avg_wait_us=sum(short_waits) / len(short_waits),
        long_avg_wait_us=sum(long_waits) / len(long_waits),
        context_switches=processor.context_switches,
        context_overhead_us=processor.context_overhead_us,
        useful_processor_utilization=useful_util,
        simulated_crash=useful_util < 70,
        notes=notes,
    )


def print_modern_result(result: ModernSchedulerResult) -> None:
    crash_label = "YES" if result.simulated_crash else "no"
    print(
        f"{result.name:22} | tasks={result.task_count:6,d} | "
        f"avg wait={result.avg_wait_us:12,.2f} us | "
        f"avg turnaround={result.avg_turnaround_us:12,.2f} us | "
        f"short wait={result.short_avg_wait_us:10,.2f} us | "
        f"long wait={result.long_avg_wait_us:12,.2f} us | "
        f"ctx={result.context_switches:9,d} | useful util={result.useful_processor_utilization:6.2f}% | "
        f"overhead crash={crash_label}"
    )
    if result.notes:
        print(f"  {result.notes}")


def print_rt_result(result: RealTimeResult) -> None:
    if result.crashed:
        print(
            f"{result.name:18} | U={result.theoretical_utilization:6.2f}% | "
            f"OVERFLOW at tick {result.crash_tick} on {result.crash_task}"
        )
    else:
        total = result.busy_ticks + result.idle_ticks
        actual = (result.busy_ticks / total) * 100 if total else 0
        print(
            f"{result.name:18} | U={result.theoretical_utilization:6.2f}% | "
            f"ok | actual busy={actual:6.2f}%"
        )


def print_table(title: str, headers: list[str], rows: list[dict[str, object]]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(no rows)")
        return

    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }
    divider = "-+-".join("-" * widths[header] for header in headers)
    header_line = " | ".join(header.ljust(widths[header]) for header in headers)

    print(header_line)
    print(divider)
    for row in rows:
        print(
            " | ".join(
                str(row.get(header, "")).ljust(widths[header])
                for header in headers
            )
        )


# ---------------------------------------------------------
# 3. HARD-REAL-TIME SCHEDULERS WITH OVERFLOW DETECTION
# ---------------------------------------------------------

def check_deadline_overflow(
    tick: int,
    tasks: list[UniversalTask],
) -> tuple[bool, UniversalTask | None]:
    for task in tasks:
        if task.current_deadline == tick and task.time_remaining > 0:
            return True, task
    return False, None


def release_periodic_jobs(tick: int, tasks: list[UniversalTask]) -> None:
    for task in tasks:
        if tick % task.period == 0:
            task.time_remaining += task.run_time
            task.current_deadline = tick + task.period


def run_rate_monotonic(
    tasks: list[UniversalTask],
    total_ticks: int = 500,
    verbose: bool = False,
) -> RealTimeResult:
    active_tasks = copy.deepcopy(tasks)
    active_tasks.sort(key=lambda task: task.period)
    cpu = SyntheticProcessor()

    for tick in range(total_ticks + 1):
        crashed, task = check_deadline_overflow(tick, active_tasks)
        if crashed:
            return RealTimeResult(
                "Rate-Monotonic",
                theoretical_utilization(tasks),
                True,
                tick,
                task.name if task else None,
                cpu.busy_ticks,
                cpu.idle_ticks,
            )

        if tick == total_ticks:
            break

        release_periodic_jobs(tick, active_tasks)
        running_task = next((t for t in active_tasks if t.time_remaining > 0), None)
        if verbose:
            print(f"Tick {tick:03d}: {running_task.name if running_task else 'IDLE'}")
        cpu.execute_tick(running_task)

    return RealTimeResult(
        "Rate-Monotonic",
        theoretical_utilization(tasks),
        False,
        None,
        None,
        cpu.busy_ticks,
        cpu.idle_ticks,
    )


def run_deadline_driven(
    tasks: list[UniversalTask],
    total_ticks: int = 500,
    verbose: bool = False,
) -> RealTimeResult:
    active_tasks = copy.deepcopy(tasks)
    cpu = SyntheticProcessor()

    for tick in range(total_ticks + 1):
        crashed, task = check_deadline_overflow(tick, active_tasks)
        if crashed:
            return RealTimeResult(
                "Deadline Driven",
                theoretical_utilization(tasks),
                True,
                tick,
                task.name if task else None,
                cpu.busy_ticks,
                cpu.idle_ticks,
            )

        if tick == total_ticks:
            break

        release_periodic_jobs(tick, active_tasks)
        ready_tasks = [t for t in active_tasks if t.time_remaining > 0]
        ready_tasks.sort(key=lambda task: task.current_deadline)
        running_task = ready_tasks[0] if ready_tasks else None
        if verbose:
            print(f"Tick {tick:03d}: {running_task.name if running_task else 'IDLE'}")
        cpu.execute_tick(running_task)

    return RealTimeResult(
        "Deadline Driven",
        theoretical_utilization(tasks),
        False,
        None,
        None,
        cpu.busy_ticks,
        cpu.idle_ticks,
    )


def run_mixed_scheduler(
    tasks: list[UniversalTask],
    total_ticks: int = 500,
    verbose: bool = False,
) -> RealTimeResult:
    active_tasks = copy.deepcopy(tasks)
    fast_tasks = sorted([t for t in active_tasks if t.period < 30], key=lambda t: t.period)
    slow_tasks = [t for t in active_tasks if t.period >= 30]
    cpu = SyntheticProcessor()

    for tick in range(total_ticks + 1):
        crashed, task = check_deadline_overflow(tick, active_tasks)
        if crashed:
            return RealTimeResult(
                "Mixed Scheduler",
                theoretical_utilization(tasks),
                True,
                tick,
                task.name if task else None,
                cpu.busy_ticks,
                cpu.idle_ticks,
            )

        if tick == total_ticks:
            break

        release_periodic_jobs(tick, active_tasks)
        running_task = next((t for t in fast_tasks if t.time_remaining > 0), None)

        if not running_task:
            ready_slow = sorted(
                [t for t in slow_tasks if t.time_remaining > 0],
                key=lambda t: t.current_deadline,
            )
            running_task = ready_slow[0] if ready_slow else None

        if verbose:
            print(f"Tick {tick:03d}: {running_task.name if running_task else 'IDLE'}")
        cpu.execute_tick(running_task)

    return RealTimeResult(
        "Mixed Scheduler",
        theoretical_utilization(tasks),
        False,
        None,
        None,
        cpu.busy_ticks,
        cpu.idle_ticks,
    )


def generate_rt_task_set(target_utilization: float, task_count: int = 3) -> list[UniversalTask]:
    # These non-harmonic periods expose Rate-Monotonic's practical upper bound.
    period_options = {
        3: [61, 78, 95],
        4: [40, 50, 70, 90],
        5: [40, 50, 70, 90, 110],
    }
    periods = period_options[task_count]

    # Keep the highest-priority task intentionally heavy so RM starts failing
    # close to the classic 3-task Liu-Layland bound of ~78%.
    shares = {
        3: [0.339, 0.469, 0.192],
        4: [0.32, 0.25, 0.23, 0.20],
        5: [0.28, 0.22, 0.20, 0.16, 0.14],
    }[task_count]

    tasks = []
    runtimes = [
        max(1, round(target_utilization * share * period))
        for period, share in zip(periods, shares)
    ]

    # Avoid accidentally asking EDF to do more than 100% useful work at the top
    # of the sweep due to integer rounding.
    while sum(c / p for c, p in zip(runtimes, periods)) > 1.0:
        index_to_trim = max(range(len(runtimes)), key=lambda i: runtimes[i] / periods[i])
        runtimes[index_to_trim] -= 1

    for index, (period, runtime) in enumerate(zip(periods, runtimes), start=1):
        tasks.append(
            UniversalTask(
                name=f"RT_Task_{index}",
                period=period,
                run_time=runtime,
            )
        )
    return tasks


def utilization_stress_tester() -> list[dict[str, object]]:
    schedulers = [
        ("Rate-Monotonic", run_rate_monotonic),
        ("Deadline Driven", run_deadline_driven),
        ("Mixed Scheduler", run_mixed_scheduler),
    ]
    summary_rows = []

    for task_count in range(3, 6):
        first_overflows: dict[str, RealTimeResult | None] = {
            name: None for name, _runner in schedulers
        }
        highest_safe_utilization: dict[str, float] = {
            name: 0.0 for name, _runner in schedulers
        }

        for target_percent in range(70, 101):
            tasks = generate_rt_task_set(target_percent / 100, task_count=task_count)
            for name, runner in schedulers:
                result = runner(tasks, total_ticks=2_000)
                if result.crashed:
                    if first_overflows[name] is None:
                        first_overflows[name] = result
                else:
                    highest_safe_utilization[name] = max(
                        highest_safe_utilization[name],
                        result.theoretical_utilization,
                    )

        for name, _runner in schedulers:
            result = first_overflows[name]
            if result:
                summary_rows.append(
                    {
                        "algorithm": name,
                        "workload": f"{task_count} RT tasks",
                        "bound": f"{result.theoretical_utilization:.2f}%",
                        "bound_value": result.theoretical_utilization,
                        "status": "OVERFLOW",
                        "detail": f"tick {result.crash_tick}, {result.crash_task}",
                    }
                )
            else:
                summary_rows.append(
                    {
                        "algorithm": name,
                        "workload": f"{task_count} RT tasks",
                        "bound": f"{highest_safe_utilization[name]:.2f}%",
                        "bound_value": highest_safe_utilization[name],
                        "status": "OK through sweep",
                        "detail": "no overflow",
                    }
                )

    return summary_rows


# ---------------------------------------------------------
# 4. MODERN / ENOKI SCHEDULERS WITH SCALABILITY METRICS
# ---------------------------------------------------------

def generate_modern_tasks(task_count: int, seed: int = 42) -> list[UniversalTask]:
    rng = random.Random(seed + task_count)
    tasks = []
    locality_options = ["Cache_A", "Cache_B", "Cache_C", "Disk_Q", "Network_Q"]
    app_groups = ["Database", "WebServer", "Analytics", "System"]

    for i in range(task_count):
        sample = rng.random()
        if sample < 0.90:
            duration_us = 10
            is_short = True
            group = "Database"
            weight = 1.0
        elif sample < 0.98:
            duration_us = rng.randint(150, 500)
            is_short = False
            group = rng.choice(app_groups)
            weight = 2.0
        else:
            duration_us = rng.randint(8_000, 20_000)
            is_short = False
            group = "Analytics"
            weight = 4.0

        tasks.append(
            UniversalTask(
                name=f"T{i:06d}",
                duration_us=duration_us,
                priority_weight=weight,
                locality_hint=rng.choice(locality_options),
                app_group=group,
                is_short=is_short,
            )
        )

    return tasks


def run_wfq(
    tasks: list[UniversalTask],
    time_slice_us: int = 100,
    context_switch_latency_us: int = 10,
) -> ModernSchedulerResult:
    active_tasks = copy.deepcopy(tasks)
    processor = SyntheticProcessor(context_switch_latency_us=context_switch_latency_us)
    heap = []
    now_us = 0

    for index, task in enumerate(active_tasks):
        heapq.heappush(heap, (task.vruntime, index, task))

    while heap:
        _vruntime, index, task = heapq.heappop(heap)
        now_us += processor.apply_dispatch_penalty(task.name)

        if task.first_run_time_us is None:
            task.first_run_time_us = now_us

        run_us = min(time_slice_us, task.duration_us)
        task.duration_us -= run_us
        now_us += run_us
        task.vruntime += run_us * task.priority_weight

        if task.duration_us > 0:
            heapq.heappush(heap, (task.vruntime, index, task))
        else:
            original = active_tasks[index]
            original.duration_us = tasks[index].duration_us
            original.completion_time_us = now_us

    return finish_modern_result("Weighted Fair Queue", active_tasks, now_us, processor)


def run_shinjuku(
    tasks: list[UniversalTask],
    micro_time_slice_us: int = 10,
    context_switch_latency_us: int = 10,
) -> ModernSchedulerResult:
    active_tasks = copy.deepcopy(tasks)
    processor = SyntheticProcessor(context_switch_latency_us=context_switch_latency_us)
    queue = deque(range(len(active_tasks)))
    now_us = 0

    while queue:
        index = queue.popleft()
        task = active_tasks[index]
        now_us += processor.apply_dispatch_penalty(task.name)

        if task.first_run_time_us is None:
            task.first_run_time_us = now_us

        run_us = min(micro_time_slice_us, task.duration_us)
        task.duration_us -= run_us
        now_us += run_us

        if task.duration_us > 0:
            queue.append(index)
        else:
            task.duration_us = tasks[index].duration_us
            task.completion_time_us = now_us

    notes = (
        "Shinjuku health check: short-query wait stays low when long jobs are sliced, "
        "but context switches grow very quickly."
    )
    return finish_modern_result("Shinjuku", active_tasks, now_us, processor, notes=notes)


def run_locality_aware(
    tasks: list[UniversalTask],
    num_cores: int = 8,
    context_switch_latency_us: int = 10,
) -> ModernSchedulerResult:
    active_tasks = copy.deepcopy(tasks)
    processor = SyntheticProcessor(context_switch_latency_us=context_switch_latency_us)
    cores: list[list[int]] = [[] for _ in range(num_cores)]
    locality_to_core: dict[str, int] = {}

    for index, task in enumerate(active_tasks):
        if task.locality_hint not in locality_to_core:
            locality_to_core[task.locality_hint] = len(locality_to_core) % num_cores
        cores[locality_to_core[task.locality_hint]].append(index)

    core_times = [0 for _ in range(num_cores)]
    for core_id, core_queue in enumerate(cores):
        for index in core_queue:
            task = active_tasks[index]
            core_times[core_id] += processor.apply_dispatch_penalty(task.name)
            task.first_run_time_us = core_times[core_id]
            core_times[core_id] += task.duration_us
            task.completion_time_us = core_times[core_id]

    makespan_us = max(core_times) if core_times else 0
    return finish_modern_result(
        "Locality Aware",
        active_tasks,
        makespan_us,
        processor,
        processor_count=num_cores,
    )


def run_arachne_arbiter(
    tasks: list[UniversalTask],
    total_system_cores: int = 8,
    context_switch_latency_us: int = 10,
) -> ModernSchedulerResult:
    active_tasks = copy.deepcopy(tasks)
    processor = SyntheticProcessor(context_switch_latency_us=context_switch_latency_us)
    app_to_tasks: dict[str, list[int]] = defaultdict(list)

    for index, task in enumerate(active_tasks):
        app_to_tasks[task.app_group].append(index)

    app_count = max(1, len(app_to_tasks))
    cores_per_app = max(1, total_system_cores // app_count)
    app_finish_times = []

    for indices in app_to_tasks.values():
        lane_times = [0 for _ in range(cores_per_app)]
        for offset, index in enumerate(indices):
            lane = offset % cores_per_app
            task = active_tasks[index]
            lane_times[lane] += processor.apply_dispatch_penalty(task.name)
            task.first_run_time_us = lane_times[lane]
            lane_times[lane] += task.duration_us
            task.completion_time_us = lane_times[lane]
        app_finish_times.append(max(lane_times) if lane_times else 0)

    makespan_us = max(app_finish_times) if app_finish_times else 0
    notes = f"Allocated {cores_per_app} core(s) per application group."
    return finish_modern_result(
        "Arachne Arbiter",
        active_tasks,
        makespan_us,
        processor,
        processor_count=total_system_cores,
        notes=notes,
    )


def modern_load_stress_tester() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    schedulers = [
        run_wfq,
        run_shinjuku,
        run_locality_aware,
        run_arachne_arbiter,
    ]
    first_degradation: dict[str, tuple[int, ModernSchedulerResult] | None] = {}
    detail_rows = []

    for task_count in [10_000, 25_000, 50_000, 100_000]:
        tasks = generate_modern_tasks(task_count)

        for runner in schedulers:
            result = runner(tasks)
            detail_rows.append(
                {
                    "algorithm": result.name,
                    "load": f"{result.task_count:,}",
                    "load_value": result.task_count,
                    "avg wait": f"{result.avg_wait_us:,.0f} us",
                    "avg_wait_value": result.avg_wait_us,
                    "avg turnaround": f"{result.avg_turnaround_us:,.0f} us",
                    "avg_turnaround_value": result.avg_turnaround_us,
                    "short wait": f"{result.short_avg_wait_us:,.0f} us",
                    "short_wait_value": result.short_avg_wait_us,
                    "long wait": f"{result.long_avg_wait_us:,.0f} us",
                    "long_wait_value": result.long_avg_wait_us,
                    "ctx switches": f"{result.context_switches:,}",
                    "ctx_switches_value": result.context_switches,
                    "useful util": f"{result.useful_processor_utilization:.2f}%",
                    "useful_util_value": result.useful_processor_utilization,
                    "status": "DEGRADED" if result.simulated_crash else "OK",
                }
            )
            first_degradation.setdefault(result.name, None)
            if result.simulated_crash and first_degradation[result.name] is None:
                first_degradation[result.name] = (task_count, result)

    summary_rows = []
    for name, entry in first_degradation.items():
        if entry:
            task_count, result = entry
            summary_rows.append(
                {
                    "algorithm": name,
                    "workload": f"{task_count:,} tasks",
                    "bound": f"{result.useful_processor_utilization:.2f}% useful util",
                    "bound_value": result.useful_processor_utilization,
                    "status": "DEGRADED",
                    "detail": "context-switch overhead bound crossed",
                }
            )
        else:
            summary_rows.append(
                {
                    "algorithm": name,
                    "workload": "100,000 tasks",
                    "bound": "not reached",
                    "bound_value": None,
                    "status": "OK through sweep",
                    "detail": "no overhead degradation",
                }
            )

    return summary_rows, detail_rows


def generate_graph_report(
    rt_summary_rows: list[dict[str, object]],
    modern_summary_rows: list[dict[str, object]],
    modern_detail_rows: list[dict[str, object]],
    output_path: str = "scheduler_stress_report.png",
) -> str:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {
        "Rate-Monotonic": "#d1495b",
        "Deadline Driven": "#00798c",
        "Mixed Scheduler": "#edae49",
        "Weighted Fair Queue": "#3066be",
        "Shinjuku": "#8f2d56",
        "Locality Aware": "#2a9d8f",
        "Arachne Arbiter": "#6a994e",
    }

    fig = plt.figure(figsize=(18, 13), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 1, 1])
    fig.suptitle(
        "Operating System Scheduler Stress Test Report",
        fontsize=22,
        fontweight="bold",
    )

    # Real-time upper bounds.
    rt_ax = fig.add_subplot(grid[0, 0])
    rt_algorithms = ["Rate-Monotonic", "Deadline Driven", "Mixed Scheduler"]
    task_counts = ["3 RT tasks", "4 RT tasks", "5 RT tasks"]
    x_positions = range(len(task_counts))
    bar_width = 0.24

    for offset, algorithm in enumerate(rt_algorithms):
        values = [
            next(
                row["bound_value"]
                for row in rt_summary_rows
                if row["algorithm"] == algorithm and row["workload"] == workload
            )
            for workload in task_counts
        ]
        shifted = [x + (offset - 1) * bar_width for x in x_positions]
        bars = rt_ax.bar(
            shifted,
            values,
            width=bar_width,
            label=algorithm,
            color=colors[algorithm],
        )
        for bar, value in zip(bars, values):
            rt_ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    rt_ax.axhline(100, color="#333333", linewidth=1, linestyle="--", alpha=0.65)
    rt_ax.set_title("Hard Real-Time Upper Bounds")
    rt_ax.set_ylabel("First overflow or safe sweep limit")
    rt_ax.set_xticks(list(x_positions))
    rt_ax.set_xticklabels(task_counts)
    rt_ax.set_ylim(0, 110)
    rt_ax.legend(fontsize=8, loc="lower right")

    # Modern useful utilization under load.
    util_ax = fig.add_subplot(grid[0, 1])
    modern_algorithms = [
        "Weighted Fair Queue",
        "Shinjuku",
        "Locality Aware",
        "Arachne Arbiter",
    ]
    for algorithm in modern_algorithms:
        rows = [row for row in modern_detail_rows if row["algorithm"] == algorithm]
        loads = [row["load_value"] for row in rows]
        values = [row["useful_util_value"] for row in rows]
        util_ax.plot(
            loads,
            values,
            marker="o",
            linewidth=2.5,
            label=algorithm,
            color=colors[algorithm],
        )
    util_ax.axhline(70, color="#d1495b", linewidth=1.5, linestyle="--", label="70% degrade line")
    util_ax.set_title("Useful Processor Utilization After Context Switch Cost")
    util_ax.set_xlabel("Task load")
    util_ax.set_ylabel("Useful utilization")
    util_ax.set_xscale("log")
    util_ax.set_ylim(0, 105)
    util_ax.legend(fontsize=8)

    # Modern average wait time.
    wait_ax = fig.add_subplot(grid[1, 0])
    for algorithm in modern_algorithms:
        rows = [row for row in modern_detail_rows if row["algorithm"] == algorithm]
        wait_ax.plot(
            [row["load_value"] for row in rows],
            [row["avg_wait_value"] / 1_000 for row in rows],
            marker="o",
            linewidth=2.5,
            label=algorithm,
            color=colors[algorithm],
        )
    wait_ax.set_title("Average Waiting Time Under Massive Load")
    wait_ax.set_xlabel("Task load")
    wait_ax.set_ylabel("Average wait, ms")
    wait_ax.set_xscale("log")
    wait_ax.legend(fontsize=8)

    # Short vs long task waiting time for Shinjuku.
    shinjuku_ax = fig.add_subplot(grid[1, 1])
    shinjuku_rows = [row for row in modern_detail_rows if row["algorithm"] == "Shinjuku"]
    loads = [row["load_value"] for row in shinjuku_rows]
    shinjuku_ax.plot(
        loads,
        [row["short_wait_value"] / 1_000 for row in shinjuku_rows],
        marker="o",
        linewidth=2.5,
        label="Short 10us tasks",
        color="#2a9d8f",
    )
    shinjuku_ax.plot(
        loads,
        [row["long_wait_value"] / 1_000 for row in shinjuku_rows],
        marker="o",
        linewidth=2.5,
        label="Medium/long tasks",
        color="#8f2d56",
    )
    shinjuku_ax.set_title("Shinjuku: Short-Task Protection vs Long-Task Delay")
    shinjuku_ax.set_xlabel("Task load")
    shinjuku_ax.set_ylabel("Average wait, ms")
    shinjuku_ax.set_xscale("log")
    shinjuku_ax.legend(fontsize=8)

    # Context switch pressure.
    ctx_ax = fig.add_subplot(grid[2, 0])
    for algorithm in modern_algorithms:
        rows = [row for row in modern_detail_rows if row["algorithm"] == algorithm]
        ctx_ax.plot(
            [row["load_value"] for row in rows],
            [row["ctx_switches_value"] for row in rows],
            marker="o",
            linewidth=2.5,
            label=algorithm,
            color=colors[algorithm],
        )
    ctx_ax.set_title("Context Switch Pressure")
    ctx_ax.set_xlabel("Task load")
    ctx_ax.set_ylabel("Context switches")
    ctx_ax.set_xscale("log")
    ctx_ax.legend(fontsize=8)

    # Status table.
    table_ax = fig.add_subplot(grid[2, 1])
    table_ax.axis("off")
    status_rows = []
    for row in rt_summary_rows + modern_summary_rows:
        status_rows.append(
            [
                row["algorithm"],
                row["workload"],
                row["bound"],
                row["status"],
            ]
        )

    table = table_ax.table(
        cellText=status_rows,
        colLabels=["Algorithm", "Workload", "Bound", "Status"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    for (row_index, _col_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#263238")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f5f7fa")
    table_ax.set_title("Upper Bound Summary", pad=14)

    absolute_path = os.path.abspath(output_path)
    fig.savefig(absolute_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return absolute_path


# ---------------------------------------------------------
# 5. EXECUTE STRESS TESTS
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Running scheduler stress tests...")
    rt_summary_rows = utilization_stress_tester()
    modern_summary_rows, modern_detail_rows = modern_load_stress_tester()
    report_path = generate_graph_report(
        rt_summary_rows,
        modern_summary_rows,
        modern_detail_rows,
    )
    print(f"Graph report saved to: {report_path}")
