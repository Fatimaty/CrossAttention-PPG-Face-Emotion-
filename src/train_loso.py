# ============== Run LOSO ==============
X_CWT, X_FACE, y, subject_ids = build_arrays_windows_layout(DATA_ROOT, use_face=USE_FACE)
print('Shapes:', X_CWT.shape, None if X_FACE is None else X_FACE.shape, y.shape, subject_ids.shape)

def chw(x): return np.transpose(x,(2,0,1))
class _DS(Dataset):
    def __init__(self, X_CWT, y, idx, X_FACE=None, use_face=True):
        self.X_CWT=X_CWT; self.X_FACE=X_FACE; self.y=y; self.idx=idx; self.use_face=use_face and (X_FACE is not None)
    def __len__(self): return len(self.idx)
    def __getitem__(self,i):
        j=self.idx[i]; cwt=chw(self.X_CWT[j]).astype('float32'); lab=int(self.y[j])
        if self.use_face:
            face=chw(self.X_FACE[j]).astype('float32'); return torch.from_numpy(cwt), torch.from_numpy(face), torch.tensor(lab)
        return torch.from_numpy(cwt), None, torch.tensor(lab)

def make_loaders(tr, va, te):
    return (
        DataLoader(_DS(X_CWT,y,tr,X_FACE,USE_FACE), batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(_DS(X_CWT,y,va,X_FACE,USE_FACE), batch_size=BATCH_SIZE),
        DataLoader(_DS(X_CWT,y,te,X_FACE,USE_FACE), batch_size=BATCH_SIZE),
    )

subjects = np.unique(subject_ids)
accs, f1s = [], []
for fold, s in enumerate(subjects,1):
    print(f"\n===== Fold {fold}/{len(subjects)} — Test subject {s} =====")
    test_idx = np.where(subject_ids==s)[0]
    train_idx = np.where(subject_ids!=s)[0]
    tr_idx, va_idx = train_test_split(train_idx, test_size=VAL_SPLIT, random_state=SEED, stratify=y[train_idx])
    dl_tr, dl_va, dl_te = make_loaders(tr_idx, va_idx, test_idx)

    cls_w = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y[tr_idx])
    cls_w = torch.tensor(cls_w, dtype=torch.float32, device=device)

    model = DualBranchCNN_XAttn_Classifier(
        cwt_in_ch=1, face_in_ch=3, use_face=USE_FACE and (X_FACE is not None),
        base_channels=BASE_CHANNELS, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
        dropout=DROPOUT, num_classes=NUM_CLASSES, pooling=POOLING
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(weight=cls_w)
    stopper = EarlyStopper(patience=PATIENCE)

    best_state=None; best_val=float('inf')
    for epoch in range(1,EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, opt, crit)
        va_loss, va_acc, va_f1, _ = eval_model(model, dl_va, crit)
        print(f"Epoch {epoch:02d} | tr {tr_loss:.4f}/{tr_acc:.3f} | va {va_loss:.4f}/{va_acc:.3f} F1 {va_f1:.3f}")
        if va_loss < best_val:
            best_val = va_loss; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        stopper.step(va_loss)
        if stopper.stop:
            print('Early stopping.'); break
    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_acc, te_f1, te_cm = eval_model(model, dl_te, crit)
    print(f"Test — loss {te_loss:.4f} acc {te_acc:.4f} macroF1 {te_f1:.4f}")
    print('CM:\n', te_cm)
    accs.append(te_acc); f1s.append(te_f1)

print('\n===== LOSO Summary =====')
print('Acc mean±std:', float(np.mean(accs)), float(np.std(accs)))
print('F1  mean±std:', float(np.mean(f1s)), float(np.std(f1s)))
