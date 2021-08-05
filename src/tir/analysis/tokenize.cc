/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*! \brief Tokenize TIR code for training
 */
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {
enum TokenType {
  Builtin = 0,
  Variable = 1,
  Number = 2,
};

std::vector<std::string> builtins = {
    "for",
    "for_min",
    "for_extent",
    "for_var",
    "for_body",
    "for_end",

    "for_kind_serial",
    "for_kind_unrolled",
    "for_kind_parallel",
    "for_kind_vectorized",

    "storage_scope",
    // Storage scope locations
    "global",  // TODO: should these have their own token type?
    "shared",
    "local",
    "warp",

    "allocate",

    "shape_start",
    "shape_end",

    "store",
    "store_predicate_start",
    "store_predicate_end",
    "store_index_start",
    "store_index_end",
    "store_value_start",
    "store_value_end",

    // dtypes
    "bool",
    "uint",
    "float",
    "int",

    "store_predicate",

    "value",
    "value_end",

    "broadcast",
    "broadcast_end",

    "store_index",
    "store_value",
    "store_end",

    "ramp",
    "ramp_base",
    "ramp_stride",
    "ramp_lanes",
    "ramp_end",

    "load",
    "load_index",
    "load_end",

    "op",
    "op_separator",
    "op_end",

    "var",

    "seq_separator",

    "add",
    "sub",
    "mul",
    "div",
    "min",
    "max",
    "mod",
    "floordiv",
    "floormod",
    "and",
    "or",
    "ne",
    "eq",
    "lt",
    "le",
    "gt",
    "ge",

    // intrinsics
    "tir.if_then_else",
    "tir.exp",
    "tir.exp2",
    "tir.exp10",
    "tir.erf",
    "tir.sigmoid",
    "tir.log",
    "tir.log2",
    "tir.log10",
    "tir.log1p",
    "tir.tan",
    "tir.tanh",
    "tir.cos",
    "tir.cosh",
    "tir.sin",
    "tir.sinh",
    "tir.asin",
    "tir.asinh",
    "tir.atan",
    "tir.atanh",
    "tir.atan2",
    "tir.sqrt",
    "tir.rsqrt",
    "tir.nextafter",
    "tir.hypot",
    "tir.copysign",
    "tir.ldexp",
    "tir.popcount",
    "tir.q_multiply_shift",
    "tir.fmod",

    "if",
    "then",
    "else",
    "if_end",
};

struct TIRTokenizer : public StmtExprVisitor {
  TIRTokenizer() {
    for (size_t i = 0; i < builtins.size(); i++) {
      builtin_tokens[builtins[i]] = i;
    }
  }

  void reset() {
    tokens.clear();
    token_types.clear();
  }

  runtime::NDArray get_output() {
    auto ary = runtime::NDArray::Empty({static_cast<int64_t>(tokens.size()), 2},
                                       runtime::DataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
    for (size_t i = 0; i < tokens.size(); i++) {
      static_cast<int64_t*>(ary->data)[2 * i] = tokens[i];
      static_cast<int64_t*>(ary->data)[2 * i + 1] = token_types[i];
    }
    return ary;
  }

  runtime::NDArray tokenize_stmt(ObjectRef o) {
    if(o.as<StmtNode>()) {
      VisitStmt(Downcast<Stmt>(o));
    } else if(o.as<PrimExprNode>()) {
      VisitExpr(Downcast<PrimExpr>(o));
    } else {
      LOG(FATAL) << "Cannot tokenize " << o->GetTypeKey();
    }
    runtime::NDArray ary = get_output();
    reset();
    return ary;
  }

  void VisitExpr_(const VarNode* op) override {
    builtin("var");
    dtype(op->dtype);
    variable(GetRef<Var>(op));
  }
  void VisitExpr_(const SizeVarNode* op) override {
    LOG(FATAL) << "Do not know how to handle SizeVar";
  }
  void VisitExpr_(const LoadNode* op) override {
    builtin("load");
    dtype(op->dtype);
    variable(op->buffer_var);
    // TODO: separator here?
    VisitExpr(op->predicate);
    builtin("load_index");
    VisitExpr(op->index);
    builtin("load_end");
  }
  void VisitExpr_(const BufferLoadNode* op) override {
    LOG(FATAL) << "Do not know how to handle BufferLoad";
  }
  void VisitExpr_(const ProducerLoadNode* op) override {
    LOG(FATAL) << "Do not know how to handle ProducerLoad";
  }
  void VisitExpr_(const LetNode* op) override { LOG(FATAL) << "Do not know how to handle Let"; }
  void VisitExpr_(const CallNode* op) override {
    if (op->op.as<OpNode>()) {
      Op o = Downcast<Op>(op->op);
      builtin("op");
      builtin(o->name);
      dtype(op->dtype);
      for (size_t i = 0; i < op->args.size(); i++) {
        VisitExpr(op->args[i]);
        if (i < op->args.size() - 1) {
          builtin("op_separator");
        }
      }
      builtin("op_end");
    } else {
      LOG(FATAL) << "Do not know how to handle Call with op of type " << op->op->GetTypeKey();
    }
  }

  template <typename Op>
  void VisitBinaryOp(const Op* op, std::string name) {
    builtin(name);
    dtype(op->dtype);
    VisitExpr(op->a);
    builtin("op_separator");
    VisitExpr(op->b);
    builtin("op_end");
  }

  void VisitExpr_(const AddNode* op) override { VisitBinaryOp(op, "add"); }
  void VisitExpr_(const SubNode* op) override { VisitBinaryOp(op, "sub"); }
  void VisitExpr_(const MulNode* op) override { VisitBinaryOp(op, "mul"); }
  void VisitExpr_(const DivNode* op) override { VisitBinaryOp(op, "div"); }
  void VisitExpr_(const ModNode* op) override { VisitBinaryOp(op, "mod"); }
  void VisitExpr_(const FloorDivNode* op) override { VisitBinaryOp(op, "floordiv"); }
  void VisitExpr_(const FloorModNode* op) override { VisitBinaryOp(op, "floormod"); }
  void VisitExpr_(const MinNode* op) override { VisitBinaryOp(op, "min"); }
  void VisitExpr_(const MaxNode* op) override { VisitBinaryOp(op, "max"); }
  void VisitExpr_(const EQNode* op) override { VisitBinaryOp(op, "eq"); }
  void VisitExpr_(const NENode* op) override { VisitBinaryOp(op, "ne"); }
  void VisitExpr_(const LTNode* op) override { VisitBinaryOp(op, "lt"); }
  void VisitExpr_(const LENode* op) override { VisitBinaryOp(op, "le"); }
  void VisitExpr_(const GTNode* op) override { VisitBinaryOp(op, "gt"); }
  void VisitExpr_(const GENode* op) override { VisitBinaryOp(op, "ge"); }
  void VisitExpr_(const AndNode* op) override { VisitBinaryOp(op, "and"); }
  void VisitExpr_(const OrNode* op) override { VisitBinaryOp(op, "or"); }
  void VisitExpr_(const ReduceNode* op) override {
    LOG(FATAL) << "Do not know how to handle Reduce";
  }
  void VisitExpr_(const CastNode* op) override { LOG(FATAL) << "Do not know how to handle Cast"; }
  void VisitExpr_(const NotNode* op) override { LOG(FATAL) << "Do not know how to handle Not"; }
  void VisitExpr_(const SelectNode* op) override {
    LOG(FATAL) << "Do not know how to handle Select";
  }
  void VisitExpr_(const RampNode* op) override {
    builtin("ramp");
    dtype(op->dtype);
    builtin("ramp_base");
    VisitExpr(op->base);
    builtin("ramp_stride");
    VisitExpr(op->stride);
    builtin("ramp_lanes");
    VisitExpr(op->lanes);
    builtin("ramp_end");
  }
  void VisitExpr_(const BroadcastNode* op) override {
    builtin("broadcast");
    dtype(op->dtype);
    numeric(op->lanes);
    VisitExpr(op->value);
    builtin("broadcast_end");
  }
  void VisitExpr_(const ShuffleNode* op) override {
    LOG(FATAL) << "Do not know how to handle Shuffle";
  }
  void VisitExpr_(const IntImmNode* op) override {
    // TODO: we only want to emit this as a value when it is being used as an index?
    builtin("value");
    dtype(op->dtype);
    numeric(op->value);
    builtin("value_end");
  }
  void VisitExpr_(const FloatImmNode* op) override {
    // TODO: Should we cast this to int?
    builtin("value");
    dtype(op->dtype);
    numeric(op->value);
    builtin("value_end");
  }
  void VisitExpr_(const StringImmNode* op) override {
    LOG(FATAL) << "Do not know how to handle StringImm";
  }
  void VisitExpr_(const AnyNode* op) override { LOG(FATAL) << "Do not know how to handle Any"; }
  void VisitStmt_(const AttrStmtNode* op) override {
    ICHECK_EQ(op->attr_key, std::string("storage_scope"))
        << "Can only handle storage_scope, got " << op->attr_key;
    builtin("storage_scope");
    variable(Downcast<Var>(op->node));
    builtin(Downcast<StringImm>(op->value)->value);
    VisitStmt(op->body);
  }
  void VisitStmt_(const IfThenElseNode* op) override {
    builtin("if");
    VisitExpr(op->condition);
    builtin("then");
    VisitStmt(op->then_case);
    if (op->else_case.defined()) {
      builtin("else");
      VisitStmt(op->else_case);
    }
    builtin("if_end");
  }
  void VisitStmt_(const LetStmtNode* op) override {
    LOG(FATAL) << "Do not know how to handle LetStmt";
  }
  void VisitStmt_(const ForNode* op) override {
    builtin("for");

    std::string for_kind;
    switch (op->kind) {
      case ForKind::kSerial:
        for_kind = "for_kind_serial";
        break;
      case ForKind::kParallel:
        for_kind = "for_kind_parallel";
        break;
      case ForKind::kVectorized:
        for_kind = "for_kind_vectorized";
        break;
      case ForKind::kUnrolled:
        for_kind = "for_kind_unrolled";
        break;
      default:
        LOG(FATAL) << "unknown for kind " << op->kind;
    }
    builtin(for_kind);

    builtin("for_min");
    VisitExpr(op->min);
    builtin("for_extent");
    VisitExpr(op->extent);
    builtin("for_var");
    variable(op->loop_var);
    builtin("for_body");
    VisitStmt(op->body);
    builtin("for_end");
  }
  void VisitStmt_(const WhileNode* op) override { LOG(FATAL) << "Do not know how to handle While"; }
  void VisitStmt_(const AllocateNode* op) override {
    builtin("allocate");
    variable(op->buffer_var);
    dtype(op->dtype);
    builtin("shape_start");
    for (size_t i = 0; i < op->extents.size(); i++) {
      VisitExpr(op->extents[i]);
      if (i < op->extents.size() - 1) {
        builtin("shape_separator");
      }
    }
    builtin("shape_end");
    VisitStmt(op->body);
  }
  void VisitStmt_(const StoreNode* op) override {
    builtin("store");
    variable(op->buffer_var);
    builtin("store_predicate");
    VisitExpr(op->predicate);
    builtin("store_index");
    VisitExpr(op->index);
    builtin("store_value");
    VisitExpr(op->value);
    builtin("store_end");
  }
  void VisitStmt_(const BufferStoreNode* op) override {
    LOG(FATAL) << "Do not know how to handle BufferStore";
  }
  void VisitStmt_(const BufferRealizeNode* op) override {
    LOG(FATAL) << "Do not know how to handle BufferRealize";
  }
  void VisitStmt_(const AssertStmtNode* op) override {
    LOG(FATAL) << "Do not know how to handle AssertStmt";
  }
  void VisitStmt_(const ProducerStoreNode* op) override {
    LOG(FATAL) << "Do not know how to handle ProducerStore";
  }
  void VisitStmt_(const ProducerRealizeNode* op) override {
    LOG(FATAL) << "Do not know how to handle ProducerRealize";
  }
  void VisitStmt_(const PrefetchNode* op) override {
    LOG(FATAL) << "Do not know how to handle Prefetch";
  }
  void VisitStmt_(const SeqStmtNode* op) override {
    for (size_t i = 0; i < op->seq.size(); i++) {
      VisitStmt(op->seq[i]);
      if (i < op->seq.size() - 1) {
        builtin("seq_separator");
      }
    }
  }
  void VisitStmt_(const EvaluateNode* op) override {
    LOG(FATAL) << "Do not know how to handle Evaluate";
  }
  void VisitStmt_(const BlockNode* op) override { LOG(FATAL) << "Do not know how to handle Block"; }
  void VisitStmt_(const BlockRealizeNode* op) override {
    LOG(FATAL) << "Do not know how to handle BlockRealize";
  }

  void emit(int64_t token, TokenType type) {
    tokens.push_back(token);
    token_types.push_back(type);
  }
  void builtin(std::string name) {
    auto it = builtin_tokens.find(name);
    if (it == builtin_tokens.end()) {
      LOG(FATAL) << "Could not find builtin " << name;
    } else {
      emit(it->second, TokenType::Builtin);
    }
  }
  void numeric(int64_t value) { emit(value, TokenType::Number); }
  void variable(Var name) {
    auto it = var_map.find(name);
    int64_t val;
    if (it != var_map.end()) {
      val = it->second;
    } else {
      val = var_map.size();
      var_map[name] = val;
    }
    emit(val, TokenType::Variable);
  }
  void dtype(DataType dt) {
    builtin(runtime::DLDataTypeCode2Str(static_cast<DLDataTypeCode>(dt.code())));
    numeric(dt.bits());
    numeric(dt.lanes());
  }
  std::map<std::string, int64_t> builtin_tokens;
  std::map<Var, int64_t> var_map;
  std::vector<int64_t> tokens;
  std::vector<int64_t> token_types;
};

TVM_REGISTER_GLOBAL("tir.analysis.tokenize").set_body_typed([](PrimFunc f) {
  // TODO: tokenize argument shapes
  TIRTokenizer t;
  t(f->body);
  auto ary = runtime::NDArray::Empty({static_cast<int64_t>(t.tokens.size()), 2},
                                     runtime::DataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  for (size_t i = 0; i < t.tokens.size(); i++) {
    static_cast<int64_t*>(ary->data)[2 * i] = t.tokens[i];
    static_cast<int64_t*>(ary->data)[2 * i + 1] = t.token_types[i];
  }
  return ary;
});

TVM_REGISTER_GLOBAL("tir.analysis.tokenize_stmt").set_body_typed([]() {
  std::shared_ptr<TIRTokenizer> t = std::make_shared<TIRTokenizer>();
  return TypedPackedFunc<runtime::NDArray(ObjectRef)>([=](ObjectRef o) {
      return t->tokenize_stmt(o);
      }, "tir.analysis.TIRTokenizer.tokenize_stmt");
});

}  // namespace tir
}  // namespace tvm
