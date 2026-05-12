# Kaycorp Review Agent TODO

## Critical

- [x] Add Task A endpoint that accepts raw user persona + product details and returns generated review + predicted rating.
- [x] Add Task B endpoint that accepts raw user persona/context and returns personalized recommendations.
- [x] Add rating prediction for unseen user-item pairs.
- [x] Add Task A metrics: RMSE and ROUGE-L.
- [x] Add Task B metrics: Hit Rate@10 and NDCG@10.
- [x] Expand solution paper to 4-8 pages.
- [ ] Verify Docker build and app startup. Blocked locally: Docker is not installed on this machine.
- [x] Commit code locally.
- [ ] Prepare GitHub remote and push.

## Important

- [x] Add cold-start recommendation workflow.
- [x] Add cross-domain recommendation evidence.
- [x] Add simple multiturn/conversational refinement endpoint.
- [x] Update README with challenge-specific endpoints and evaluation commands.
- [x] Add model/data disclosure section.

## Nice To Have

- [ ] Add BERTScore if dependency budget allows.
- [ ] Add presentation deck outline.
- [ ] Add screenshots of API docs and Streamlit demo.
