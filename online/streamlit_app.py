"""Human-in-the-loop Streamlit review and official AIC26 submission surface."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import streamlit as st

from shared.schemas.online import (
    AnswerTarget,
    KISCandidate,
    QACandidate,
    QueryRole,
    QuerySpec,
    QueryUnit,
    SearchRequest,
    TaskType,
    TrakeCandidate,
    UnifiedQueryPlan,
)

from online.engine import OnlineEngine
from online.media import evidence_image, source_video
from online.query_bundle import load_query_specs_from_zip
from online.refinement import ExactFrameDecoder
from online.workspace import SubmissionWorkspace


st.set_page_config(page_title="LASTDANCE Online", layout="wide")


@st.cache_resource(show_spinner="Loading and validating production artifacts...")
def _engine() -> OnlineEngine:
    return OnlineEngine.from_environment(deep_preflight=False)


def _split_editor_values(value: object) -> list[str]:
    return list(
        dict.fromkeys(
            item.strip()
            for item in str(value or "").replace(";", "|").split("|")
            if item.strip()
        )
    )


def _optional_int(value: object) -> int | None:
    text = str(value or "").strip()
    if not text or text.casefold() in {"none", "nan"}:
        return None
    return int(float(text))


def _render_query_plan_editor(
    plan: UnifiedQueryPlan,
    spec: QuerySpec,
) -> UnifiedQueryPlan | None:
    st.subheader("Phân tích truy vấn")
    st.caption(
        "Một unit có thể mang nhiều role. Dùng dấu | để phân tách roles, modalities và text clues."
    )
    for warning in plan.planner_warnings:
        st.warning(warning)
    global_context = st.text_area(
        "Global context (English)",
        value=plan.global_context_en,
        height=90,
        key=f"plan-global-{spec.query_name}",
    )
    rows = [
        {
            "unit_id": unit.unit_id,
            "description_original": unit.description_original,
            "retrieval_query_en": unit.retrieval_query_en,
            "roles": " | ".join(role.value for role in unit.roles),
            "requiredness": unit.requiredness,
            "modalities": " | ".join(unit.modalities),
            "temporal_group": unit.temporal_group,
            "temporal_order": unit.temporal_order,
            "known_text_literals": " | ".join(unit.known_text_literals),
            "visual_text_attributes": " | ".join(unit.visual_text_attributes),
            "confidence": unit.confidence,
        }
        for unit in plan.query_units
    ]
    edited_rows = st.data_editor(
        rows,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=["unit_id", "description_original", "confidence"],
        key=f"plan-units-{spec.query_name}",
    )
    records = edited_rows.to_dict("records") if hasattr(edited_rows, "to_dict") else edited_rows
    try:
        units = [
            QueryUnit(
                unit_id=str(row["unit_id"]),
                description_original=str(row["description_original"]),
                retrieval_query_en=str(row["retrieval_query_en"]),
                roles=[QueryRole(value.strip().upper()) for value in _split_editor_values(row["roles"])],
                requiredness=str(row["requiredness"]).strip().lower(),
                modalities=[value.lower() for value in _split_editor_values(row["modalities"])],
                temporal_group=_optional_int(row.get("temporal_group")),
                temporal_order=_optional_int(row.get("temporal_order")),
                known_text_literals=_split_editor_values(row.get("known_text_literals")),
                visual_text_attributes=_split_editor_values(row.get("visual_text_attributes")),
                confidence=float(row["confidence"]),
            )
            for row in records
        ]
    except Exception as error:
        st.error(f"Query unit edit is invalid: {error}")
        return None

    unit_ids = [unit.unit_id for unit in units]
    answer_target = plan.answer_target
    ordered_event_ids: list[str] = []
    submission_target_ids: list[str] = []
    if spec.task_type == TaskType.KIS:
        submission_target_ids = st.multiselect(
            "KIS submission target units",
            options=unit_ids,
            default=[value for value in plan.submission_target_ids if value in unit_ids],
            key=f"plan-targets-{spec.query_name}",
        )
        units = [
            unit.model_copy(
                update={
                    "roles": list(
                        dict.fromkeys(
                            [
                                *unit.roles,
                                *(
                                    [QueryRole.TARGET_MOMENT]
                                    if unit.unit_id in submission_target_ids
                                    else []
                                ),
                            ]
                        )
                    )
                }
            )
            for unit in units
        ]
        answer_target = None
    elif spec.task_type == TaskType.QA:
        current_evidence = answer_target.evidence_unit_ids if answer_target else []
        evidence_ids = st.multiselect(
            "QA answer-evidence units",
            options=unit_ids,
            default=[value for value in current_evidence if value in unit_ids],
            key=f"plan-evidence-{spec.query_name}",
        )
        source = st.selectbox(
            "Unknown answer source",
            ["visual", "ocr", "asr", "mixed"],
            index=["visual", "ocr", "asr", "mixed"].index(
                answer_target.source if answer_target else "visual"
            ),
            key=f"plan-answer-source-{spec.query_name}",
        )
        value_type = st.selectbox(
            "Unknown answer type",
            ["number", "color", "person", "place", "free_text"],
            index=["number", "color", "person", "place", "free_text"].index(
                answer_target.value_type if answer_target else "free_text"
            ),
            key=f"plan-answer-type-{spec.query_name}",
        )
        question = st.text_area(
            "Question passed to OCR/VLM answerer",
            value=answer_target.question if answer_target else spec.raw_query,
            height=90,
            key=f"plan-question-{spec.query_name}",
        )
        units = [
            unit.model_copy(
                update={
                    "roles": list(
                        dict.fromkeys(
                            [
                                *unit.roles,
                                *(
                                    [QueryRole.TARGET_MOMENT, QueryRole.ANSWER_EVIDENCE]
                                    if unit.unit_id in evidence_ids
                                    else []
                                ),
                            ]
                        )
                    ),
                    "modalities": list(
                        dict.fromkeys(
                            [
                                *unit.modalities,
                                *(["ocr"] if unit.unit_id in evidence_ids and source in {"ocr", "mixed"} else []),
                                *(["asr"] if unit.unit_id in evidence_ids and source in {"asr", "mixed"} else []),
                            ]
                        )
                    ),
                }
            )
            for unit in units
        ]
        answer_target = AnswerTarget(
            question=question,
            value_type=value_type,
            source=source,
            evidence_unit_ids=evidence_ids,
            value_is_unknown=True,
        ) if evidence_ids else None
        submission_target_ids = evidence_ids
    else:
        ordered_text = st.text_input(
            "TRAKE ordered event IDs (chronological, separated by |)",
            value=" | ".join(plan.ordered_event_ids),
            key=f"plan-events-{spec.query_name}",
        )
        ordered_event_ids = _split_editor_values(ordered_text)
        ordered_set = set(ordered_event_ids)
        units = [
            unit.model_copy(
                update={
                    "roles": list(
                        dict.fromkeys(
                            [
                                *(role for role in unit.roles if role != QueryRole.ORDERED_EVENT),
                                *(
                                    [QueryRole.ORDERED_EVENT]
                                    if unit.unit_id in ordered_set
                                    else []
                                ),
                            ]
                        )
                    )
                }
            )
            for unit in units
        ]
        answer_target = None

    try:
        draft = UnifiedQueryPlan.model_validate(
            {
                **plan.model_dump(mode="python"),
                "global_context_en": global_context,
                "caption_en": global_context,
                "query_units": [unit.model_dump(mode="python") for unit in units],
                "answer_target": answer_target.model_dump(mode="python") if answer_target else None,
                "ordered_event_ids": ordered_event_ids,
                "submission_target_ids": submission_target_ids,
                "operator_reviewed": True,
            }
        )
        before = plan.model_dump(exclude={"operator_reviewed", "operator_edited"})
        after = draft.model_dump(exclude={"operator_reviewed", "operator_edited"})
        draft = draft.model_copy(update={"operator_edited": before != after})
        return draft.validate_for_task(
            spec.task_type,
            expected_event_count=spec.expected_event_count,
        )
    except Exception as error:
        st.error(f"Query plan is not ready: {error}")
        return None


def _candidate_summary(candidate: object) -> str:
    if isinstance(candidate, TrakeCandidate):
        return " → ".join(str(value) for value in candidate.frame_ids)
    if isinstance(candidate, QACandidate):
        return f"frame {candidate.frame_id} · {candidate.answer}"
    assert isinstance(candidate, KISCandidate)
    return f"frame {candidate.frame_id}"


def _candidate_rows(candidates: list[object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, candidate in enumerate(candidates, 1):
        if isinstance(candidate, TrakeCandidate):
            rows.append(
                {
                    "rank": rank,
                    "video_id": candidate.video_id,
                    "frame_ids": " → ".join(map(str, candidate.frame_ids)),
                    "score": round(candidate.score, 6),
                    "matched_scene": " | ".join(
                        dict.fromkeys(frame.query_part for frame in candidate.evidence if frame.query_part)
                    ),
                }
            )
        else:
            evidence = candidate.evidence
            row = {
                "rank": rank,
                "video_id": candidate.video_id,
                "frame_id": candidate.frame_id,
                "pts_time": round(evidence.pts_time, 3),
                "shot_id": evidence.shot_id,
                "score": round(candidate.score, 6),
                "matched_scene": evidence.query_part,
            }
            if isinstance(candidate, QACandidate):
                row["answer"] = candidate.answer
                row["confidence"] = round(candidate.confidence, 4)
                row["requires_review"] = candidate.requires_review
            rows.append(row)
    return rows


def _render_top_candidates(engine: OnlineEngine, candidates: list[object], query_name: str) -> None:
    st.dataframe(_candidate_rows(candidates), use_container_width=True, hide_index=True)
    if not candidates:
        st.info("Không có candidate hợp lệ.")
        return
    page_count = max(1, (len(candidates) + 24) // 25)
    page = st.number_input(
        "Trang thumbnail",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
        key=f"top-page-{query_name}",
    )
    start = (int(page) - 1) * 25
    page_values = candidates[start : start + 25]
    columns = st.columns(5)
    for offset, candidate in enumerate(page_values):
        with columns[offset % 5]:
            evidence = candidate.evidence[0] if isinstance(candidate, TrakeCandidate) else candidate.evidence
            image = evidence_image(engine.registry, evidence.keyframe_uid)
            if image:
                st.image(
                    str(image),
                    caption=f"#{start + offset + 1} · {candidate.video_id} · {evidence.frame_id}",
                    use_container_width=True,
                )
    inspect_rank = st.selectbox(
        "Xem chi tiết một kết quả",
        options=list(range(1, len(candidates) + 1)),
        key=f"inspect-rank-{query_name}",
    )
    candidate = candidates[int(inspect_rank) - 1]
    evidence_values = candidate.evidence if isinstance(candidate, TrakeCandidate) else [candidate.evidence]
    detail_columns = st.columns(min(4, len(evidence_values)))
    for index, evidence in enumerate(evidence_values):
        with detail_columns[index % len(detail_columns)]:
            image = evidence_image(engine.registry, evidence.keyframe_uid)
            if image:
                st.image(str(image), caption=f"frame {evidence.frame_id} · {evidence.pts_time:.2f}s")
            neighbors = engine.registry.catalog.neighbors(evidence.keyframe_uid, 2)
            st.caption(
                "Neighbors: "
                + ", ".join(f"{item.frame_id}@{item.pts_time:.2f}s" for item in neighbors)
            )


def _render_evidence(
    engine: OnlineEngine,
    candidate: object,
    index: int,
    query_name: str,
) -> object:
    prefix = f"{query_name}-{index}"
    selected = st.checkbox("Chọn vào draft", key=f"candidate-{prefix}")
    if isinstance(candidate, TrakeCandidate):
        columns = st.columns(min(len(candidate.evidence), 4))
        edited_evidence = []
        for evidence_index, evidence in enumerate(candidate.evidence):
            catalog_choices = [engine.registry.catalog.by_uid[evidence.keyframe_uid]]
            catalog_choices.extend(engine.registry.catalog.neighbors(evidence.keyframe_uid, 2))
            unique_choices = {item.keyframe_uid: item for item in catalog_choices}
            with columns[evidence_index % len(columns)]:
                selected_uid = st.selectbox(
                    f"Moment {evidence_index + 1}",
                    options=list(unique_choices),
                    format_func=lambda uid: (
                        f"frame {unique_choices[uid].frame_id} · {unique_choices[uid].pts_time:.2f}s"
                    ),
                    key=f"trake-neighbor-{prefix}-{evidence_index}",
                )
                selected_frame = unique_choices[selected_uid]
                edited = evidence.model_copy(
                    update={
                        "keyframe_uid": selected_frame.keyframe_uid,
                        "frame_id": selected_frame.frame_id,
                        "pts_time": selected_frame.pts_time,
                        "shot_id": selected_frame.shot_id,
                    }
                )
                edited_evidence.append(edited)
                image = evidence_image(engine.registry, selected_uid)
                if image:
                    st.image(str(image), caption=f"#{edited.frame_id} · {edited.pts_time:.2f}s")
        try:
            edited_candidate = TrakeCandidate.model_validate(
                {
                    **candidate.model_dump(),
                    "frame_ids": [item.frame_id for item in edited_evidence],
                    "pts_times": [item.pts_time for item in edited_evidence],
                    "evidence": [item.model_dump() for item in edited_evidence],
                }
            )
        except Exception as error:
            st.error(f"TRAKE neighbor selection is invalid: {error}")
            return None
        return edited_candidate if selected else None

    evidence = candidate.evidence
    left, right = st.columns([1, 2])
    image = evidence_image(engine.registry, evidence.keyframe_uid)
    with left:
        if image:
            st.image(str(image), caption=f"{candidate.video_id} · {evidence.pts_time:.2f}s")
    with right:
        st.caption(
            f"score={candidate.score:.4f} · visual={evidence.score_visual:.4f} · "
            f"neighbor={evidence.neighbor_support:.4f} · consensus={evidence.model_consensus:.2f} · "
            f"vlm={evidence.vlm_score if evidence.vlm_score is not None else '—'}"
        )
        frame_id = int(
            st.number_input(
                "Submission frame_id",
                min_value=0,
                value=int(candidate.frame_id),
                step=1,
                key=f"frame-{prefix}",
            )
        )
        if isinstance(candidate, QACandidate):
            if candidate.requires_review:
                st.warning("Answer này chưa qua auto-accept; chọn thủ công đồng nghĩa bạn đã xác minh.")
            answer = st.text_input(
                "Answer (max 100 characters)",
                value=candidate.answer,
                max_chars=100,
                key=f"answer-{prefix}",
            )
            edited = candidate.model_copy(
                update={
                    "frame_id": frame_id,
                    "answer": answer,
                    "requires_review": False if selected else candidate.requires_review,
                }
            )
        else:
            edited = candidate.model_copy(update={"frame_id": frame_id})
        if st.button("Decode exact source frame", key=f"decode-{prefix}"):
            try:
                with st.spinner("FFmpeg is decoding by exact frame index..."):
                    exact = ExactFrameDecoder(engine.registry).decode(candidate.video_id, frame_id)
                st.image(str(exact), caption=f"Exact source frame #{frame_id}")
            except Exception as error:
                st.error(str(error))
    return edited if selected else None


def _render_grouped(
    engine: OnlineEngine,
    run: object,
    query_name: str,
) -> list[object]:
    grouped: dict[str, list[tuple[int, object]]] = defaultdict(list)
    for index, candidate in enumerate(run.top_candidates):
        grouped[candidate.video_id].append((index, candidate))
    selected: list[object] = []
    for video in run.video_hypotheses:
        values = grouped.get(video.video_id)
        if not values:
            continue
        with st.expander(
            f"{video.video_id} · score {video.video_score:.4f} · coverage {video.coverage:.2f} "
            f"· VLM {'yes' if video.vlm_verified else 'no'}",
            expanded=video == run.video_hypotheses[0],
        ):
            st.caption(f"Matched: {video.matched_scenes or '—'} · Missing: {video.missing_scenes or '—'}")
            source = source_video(engine.registry, video.video_id)
            if source and st.checkbox("Show source video", key=f"video-{query_name}-{video.video_id}"):
                first = values[0][1]
                start = int(first.evidence[0].pts_time if isinstance(first, TrakeCandidate) else first.evidence.pts_time)
                st.video(str(source), start_time=start)
            for index, candidate in values:
                st.markdown(f"**#{index + 1} · {_candidate_summary(candidate)}**")
                edited = _render_evidence(engine, candidate, index, query_name)
                if edited is not None:
                    selected.append(edited)
                st.divider()
    return selected


def _valid_bulk(candidates: list[object], spec: QuerySpec) -> list[object]:
    result = []
    for candidate in candidates:
        if isinstance(candidate, QACandidate):
            if not candidate.answer.strip() or candidate.answer.strip().casefold() == "uncertain":
                continue
            if candidate.requires_review:
                continue
        if isinstance(candidate, TrakeCandidate):
            if len(candidate.frame_ids) != spec.expected_event_count:
                continue
        result.append(candidate)
    return result[:100]


def _workspace(
    engine: OnlineEngine,
    specs: list[QuerySpec],
    folder: str,
) -> SubmissionWorkspace:
    current = st.session_state.setdefault("query_drafts", {})
    drafts = {spec.query_name: current.get(spec.query_name, []) for spec in specs}
    return SubmissionWorkspace(
        folder_name=folder,
        expected_queries=specs,
        layout=engine.registry.layout,
        query_drafts=drafts,
        query_history=st.session_state.setdefault("draft_history", []),
        provenance=st.session_state.setdefault("draft_provenance", {}),
        catalog=engine.registry.catalog,
    )


def _store_workspace(workspace: SubmissionWorkspace) -> None:
    st.session_state["query_drafts"] = workspace.query_drafts
    workspace.save()


def _workspace_controls(
    engine: OnlineEngine,
    spec: QuerySpec,
    specs: list[QuerySpec],
    selected: list[object],
    ranked: list[object],
) -> None:
    st.subheader("Submission workspace")
    folder = st.text_input("Workspace folder", value="submission-review")
    zip_name = st.text_input("Official ZIP name", value="submissionround1.zip")
    workspace = _workspace(engine, specs, folder)
    draft = workspace.query_drafts[spec.query_name]
    valid_bulk = _valid_bulk(ranked, spec)
    mode = st.radio(
        "Khi draft hiện tại đã có dữ liệu",
        ["Thay thế draft", "Giữ draft và điền phần còn trống"],
        horizontal=True,
        key=f"bulk-mode-{spec.query_name}",
    )
    if spec.task_type == TaskType.KIS:
        bulk_label = f"Thêm Top {len(valid_bulk)} frame vào CSV"
    elif spec.task_type == TaskType.QA:
        bulk_label = f"Thêm {len(valid_bulk)} kết quả QA có answer hợp lệ"
    else:
        bulk_label = f"Thêm Top {len(valid_bulk)} chuỗi TRAKE hợp lệ"
    add_selected, add_bulk = st.columns(2)
    with add_selected:
        if st.button(
            "Thêm các hypothesis đã chọn",
            disabled=not selected,
            key=f"add-selected-{spec.query_name}",
        ):
            try:
                workspace.merge_ranked(spec.query_name, selected)
                _store_workspace(workspace)
                st.success(f"Draft {spec.query_name}: {len(workspace.query_drafts[spec.query_name])} dòng.")
                st.rerun()
            except Exception as error:
                st.error(str(error))
    with add_bulk:
        if st.button(
            bulk_label,
            type="primary",
            disabled=not valid_bulk,
            key=f"add-bulk-{spec.query_name}",
        ):
            try:
                if draft and mode == "Giữ draft và điền phần còn trống":
                    workspace.merge_ranked(spec.query_name, valid_bulk)
                else:
                    workspace.replace_query_draft(spec.query_name, valid_bulk)
                run = st.session_state.get("search_run")
                if run is not None:
                    workspace.query_history.append(
                        {
                            "query_name": spec.query_name,
                            "request": run.request.model_dump(mode="json"),
                            "query_plan": run.query_plan.model_dump(mode="json"),
                            "timings_ms": run.timings_ms,
                            "warnings": run.warnings,
                        }
                    )
                    workspace.provenance.update(run.provenance)
                _store_workspace(workspace)
                st.success(f"Draft {spec.query_name}: {len(workspace.query_drafts[spec.query_name])} dòng.")
                st.rerun()
            except Exception as error:
                st.error(str(error))

    draft = st.session_state.get("query_drafts", {}).get(spec.query_name, [])
    progress = {
        item.query_name: len(st.session_state.get("query_drafts", {}).get(item.query_name, []))
        for item in specs
    }
    st.caption(
        f"Tiến độ bundle: {sum(count > 0 for count in progress.values())}/{len(specs)} query có draft."
    )
    if not draft:
        st.info("Draft của query này đang rỗng.")
    else:
        draft_rows = [
            {
                "position": index + 1,
                "video_id": item.video_id,
                "frame_id": item.frame_id if isinstance(item, (KISCandidate, QACandidate)) else None,
                "answer": item.answer if isinstance(item, QACandidate) else None,
                "sequence": " → ".join(map(str, item.frame_ids)) if isinstance(item, TrakeCandidate) else None,
                "score": round(item.score, 5),
            }
            for index, item in enumerate(draft)
        ]
        edited_rows = st.data_editor(
            draft_rows,
            use_container_width=True,
            hide_index=True,
            disabled=["position", "video_id", "sequence", "score"],
            num_rows="fixed",
            key=f"draft-editor-{spec.query_name}",
        )
        if spec.task_type in {TaskType.KIS, TaskType.QA} and st.button(
            "Apply draft edits",
            key=f"apply-edits-{spec.query_name}",
        ):
            try:
                records = edited_rows.to_dict("records") if hasattr(edited_rows, "to_dict") else edited_rows
                replacements = []
                for item, row in zip(draft, records):
                    updates = {"frame_id": int(row["frame_id"])}
                    if isinstance(item, QACandidate):
                        updates["answer"] = str(row["answer"] or "")
                    replacements.append(item.model_copy(update=updates))
                workspace.replace_query_draft(spec.query_name, replacements)
                _store_workspace(workspace)
                st.success("Draft edits validated and applied.")
                st.rerun()
            except Exception as error:
                st.error(f"Draft edit blocked: {error}")
        remove = st.multiselect(
            "Rows to remove",
            options=list(range(1, len(draft) + 1)),
            key=f"remove-{spec.query_name}",
        )
        if st.button(
            "Remove selected rows",
            disabled=not remove,
            key=f"remove-action-{spec.query_name}",
        ):
            workspace.remove(spec.query_name, [value - 1 for value in remove])
            _store_workspace(workspace)
            st.rerun()
        if st.button("Validate and download current CSV for inspection", key=f"csv-{spec.query_name}"):
            try:
                csv_path = workspace.export_csv(spec.query_name)
                st.success(
                    f"PASS · {len(draft)} rows · no header · UTF-8 · {spec.csv_filename}"
                )
                st.download_button(
                    "Download inspection CSV (not the final submission)",
                    csv_path.read_bytes(),
                    file_name=csv_path.name,
                    mime="text/csv",
                    key=f"csv-download-{spec.query_name}",
                )
            except Exception as error:
                st.error(f"CSV export blocked: {error}")

    st.markdown("#### Final bundle")
    if st.button("Validate all queries and build official ZIP", key="official-zip"):
        try:
            workspace = _workspace(engine, specs, folder)
            report = workspace.export_zip(zip_name=zip_name)
            st.success(
                f"PASS · {len(report.row_counts)} CSV · ZIP SHA-256 {report.zip_sha256}"
            )
            st.json(
                {
                    "profile": "AIC26_QUALIFIER_OFFICIAL",
                    "row_counts": report.row_counts,
                    "csv_sha256": report.csv_sha256,
                    "zip_sha256": report.zip_sha256,
                }
            )
            st.download_button(
                "Download official submission ZIP",
                report.zip_path.read_bytes(),
                file_name=report.zip_path.name,
                mime="application/zip",
                key="official-zip-download",
            )
        except Exception as error:
            st.error(f"Official ZIP blocked: {error}")


def _query_selector() -> tuple[QuerySpec, list[QuerySpec]]:
    uploaded = st.file_uploader("Gói query chính thức (.zip)", type=["zip"])
    if uploaded is not None:
        payload = uploaded.getvalue()
        fingerprint = hashlib.sha256(payload).hexdigest()
        if st.session_state.get("query_bundle_sha") != fingerprint:
            specs = load_query_specs_from_zip(payload)
            st.session_state["query_bundle_sha"] = fingerprint
            st.session_state["query_specs"] = specs
            existing = st.session_state.get("query_drafts", {})
            st.session_state["query_drafts"] = {
                spec.query_name: existing.get(spec.query_name, []) for spec in specs
            }
        specs = st.session_state["query_specs"]
        selected_name = st.selectbox(
            "Query",
            options=[item.query_name for item in specs],
        )
        spec = next(item for item in specs if item.query_name == selected_name)
        st.text_area("Query content", value=spec.raw_query, height=150, disabled=True)
        if spec.task_type == TaskType.TRAKE:
            st.info(f"TRAKE yêu cầu chính xác {spec.expected_event_count} frame IDs mỗi dòng.")
        return spec, specs

    task_type = TaskType(st.selectbox("Task", [item.value for item in TaskType]))
    default_name = f"query-manual-{task_type.value.lower()}"
    query_name = st.text_input("Query name", value=default_name)
    query = st.text_area("Query", height=120, placeholder="Describe the target moment...")
    expected = None
    if task_type == TaskType.TRAKE:
        expected = int(st.number_input("Expected TRAKE event count", min_value=2, value=2, step=1))
    if not query.strip():
        st.info("Nhập query hoặc upload gói query chính thức để bắt đầu.")
        st.stop()
    spec = QuerySpec(
        query_name=query_name,
        source_filename=f"{query_name}.txt",
        task_type=task_type,
        raw_query=query,
        expected_event_count=expected,
    )
    return spec, [spec]


def main() -> None:
    st.title("LASTDANCE · Accuracy-Max Online")
    try:
        engine = _engine()
    except Exception as error:
        st.error(f"NOT_READY: {type(error).__name__}: {error}")
        st.stop()

    with st.sidebar:
        st.header("Artifact preflight")
        for name, status in engine.registry.statuses.items():
            marker = "✅" if status.availability.value == "READY" else "⚠️"
            st.caption(f"{marker} {name}: {status.availability.value} · {status.detail}")
        st.caption("OCR/ASR unavailable does not block visual-only retrieval.")
        st.caption("Submission profile: AIC26_QUALIFIER_OFFICIAL")

    try:
        spec, specs = _query_selector()
    except Exception as error:
        st.error(f"Query package invalid: {error}")
        return
    max_results = st.slider("Maximum hypotheses", min_value=1, max_value=100, value=100)
    plan_store = st.session_state.setdefault("query_plans", {})
    analyze, rerun_search = st.columns(2)
    with analyze:
        analyze_clicked = st.button(
            "1. Phân tích truy vấn",
            type="primary" if spec.query_name not in plan_store else "secondary",
            key=f"analyze-{spec.query_name}",
        )
    if analyze_clicked:
        try:
            with st.spinner("Gemini/Qwen đang tách locator, target và task evidence..."):
                plan_store[spec.query_name] = engine.plan(spec)
                current_run = st.session_state.get("search_run")
                if (
                    current_run is not None
                    and current_run.request.query_spec is not None
                    and current_run.request.query_spec.query_name == spec.query_name
                ):
                    st.session_state.pop("search_run", None)
        except Exception as error:
            st.error(f"Planning failed: {type(error).__name__}: {error}")

    generated_plan = plan_store.get(spec.query_name)
    if generated_plan is None:
        st.info("Bấm ‘Phân tích truy vấn’ để xem và xác nhận vai trò trước khi retrieval.")
        _workspace_controls(engine, spec, specs, [], [])
        return
    edited_plan = _render_query_plan_editor(generated_plan, spec)
    with rerun_search:
        search_clicked = st.button(
            "2. Chạy retrieval với plan đã duyệt",
            type="primary",
            disabled=edited_plan is None,
            key=f"search-{spec.query_name}",
        )
    if search_clicked and edited_plan is not None:
        try:
            plan_store[spec.query_name] = edited_plan
            with st.spinner("Retrieving and verifying role-aware frame evidence..."):
                st.session_state["search_run"] = engine.search(
                    SearchRequest(
                        task_type=spec.task_type,
                        raw_query=spec.raw_query,
                        query_spec=spec,
                        max_results=max_results,
                    ),
                    query_plan=edited_plan,
                )
        except Exception as error:
            st.error(f"Search failed: {type(error).__name__}: {error}")

    run = st.session_state.get("search_run")
    if (
        run is None
        or run.request.query_spec is None
        or run.request.query_spec.query_name != spec.query_name
    ):
        _workspace_controls(engine, spec, specs, [], [])
        return
    for warning in run.warnings:
        st.warning(warning)
    with st.expander("Query plan, timing and provenance"):
        st.json(run.query_plan.model_dump(mode="json"))
        st.json({"timings_ms": run.timings_ms, "provenance": run.provenance})

    ranked = list(run.top_candidates)
    top_tab, video_tab = st.tabs(["Top 100", "Theo video"])
    with top_tab:
        st.subheader(f"Top {len(ranked)} submission hypotheses")
        _render_top_candidates(engine, ranked, spec.query_name)
    with video_tab:
        selected = _render_grouped(engine, run, spec.query_name)
    _workspace_controls(engine, spec, specs, selected, ranked)


if __name__ == "__main__":
    main()
