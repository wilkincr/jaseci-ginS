"""Lark parser for Jac Lang."""
from __future__ import annotations


import logging
import os
from typing import Callable, TypeAlias


import jaclang.compiler.absyntree as ast
from jaclang.compiler import jac_lark as jl  # type: ignore
from jaclang.compiler.constant import EdgeDir, Tokens as Tok
from jaclang.compiler.passes.ir_pass import Pass
from jaclang.vendor.lark import Lark, Transformer, Tree, logger


class JacParser(Pass):
    """Jac Parser."""

    dev_mode = False

    def __init__(self, input_ir: ast.JacSource) -> None:
        """Initialize parser."""
        self.source = input_ir
        self.mod_path = input_ir.loc.mod_path
        if JacParser.dev_mode:
            JacParser.make_dev()
        Pass.__init__(self, input_ir=input_ir, prior=None)

    def transform(self, ir: ast.AstNode) -> ast.AstNode:
        """Transform input IR."""
        try:
            tree, comments = JacParser.parse(
                self.source.value, on_error=self.error_callback
            )
            mod = JacParser.TreeToAST(parser=self).transform(tree)
            self.source.comments = [self.proc_comment(i, mod) for i in comments]
        except jl.UnexpectedInput as e:
            catch_error = ast.EmptyToken()
            catch_error.file_path = self.mod_path
            catch_error.line_no = e.line
            catch_error.c_start = e.column
            catch_error.c_end = e.column
            self.error(f"Syntax Error: {e}", node_override=catch_error)
            mod = self.source
        except Exception as e:
            mod = self.source
            self.error(f"Internal Error: {e}")
        return mod

    @staticmethod
    def proc_comment(token: jl.Token, mod: ast.AstNode) -> ast.CommentToken:
        """Process comment."""
        return ast.CommentToken(
            file_path=mod.loc.mod_path,
            name=token.type,
            value=token.value,
            line=token.line if token.line is not None else 0,
            col_start=token.column if token.column is not None else 0,
            col_end=token.end_column if token.end_column is not None else 0,
            pos_start=token.start_pos if token.start_pos is not None else 0,
            pos_end=token.end_pos if token.end_pos is not None else 0,
            kid=[],
        )

    def error_callback(self, e: jl.UnexpectedInput) -> bool:
        """Handle error."""
        return False

    @staticmethod
    def _comment_callback(comment: jl.Token) -> None:
        JacParser.comment_cache.append(comment)

    @staticmethod
    def parse(
        ir: str, on_error: Callable[[jl.UnexpectedInput], bool]
    ) -> tuple[jl.Tree[jl.Tree[str]], list[jl.Token]]:
        """Parse input IR."""
        JacParser.comment_cache = []
        return (
            JacParser.parser.parse(ir, on_error=on_error),
            JacParser.comment_cache,
        )

    @staticmethod
    def make_dev() -> None:
        """Make parser in dev mode."""
        JacParser.parser = Lark.open(
            "jac.lark",
            parser="lalr",
            rel_to=__file__,
            debug=True,
            lexer_callbacks={"COMMENT": JacParser._comment_callback},
        )
        JacParser.JacTransformer = Transformer[Tree[str], ast.AstNode]  # type: ignore
        logger.setLevel(logging.DEBUG)

    comment_cache: list[jl.Token] = []

    parser = jl.Lark_StandAlone(lexer_callbacks={"COMMENT": _comment_callback})  # type: ignore
    JacTransformer: TypeAlias = jl.Transformer[jl.Tree[str], ast.AstNode]

    class TreeToAST(JacTransformer):
        """Transform parse tree to AST."""

        def __init__(self, parser: JacParser, *args: bool, **kwargs: bool) -> None:
            """Initialize transformer."""
            super().__init__(*args, **kwargs)
            self.parse_ref = parser

        def ice(self) -> Exception:
            """Raise internal compiler error."""
            self.parse_ref.error("Internal Compiler Error, Invalid Parse Tree!")
            return RuntimeError(
                f"{self.parse_ref.__class__.__name__} - Internal Compiler Error, Invalid Parse Tree!"
            )

        def nu(self, node: ast.T) -> ast.T:
            """Update node."""
            self.parse_ref.cur_node = node
            return node

        def start(self, kid: list[ast.Module]) -> ast.Module:
            """Grammar rule.

            start: module
            """
            return self.nu(kid[0])

        def module(
            self, kid: list[ast.ElementStmt | ast.String | ast.EmptyToken]
        ) -> ast.Module:
            """Grammar rule.

            module: (doc_tag? element (element_with_doc | element)*)?
            doc_tag (element_with_doc (element_with_doc | element)*)?
            """
            doc = kid[0] if len(kid) and isinstance(kid[0], ast.String) else None
            body = kid[1:] if doc else kid
            body = [i for i in body if isinstance(i, ast.ElementStmt)]
            mod = ast.Module(
                name=self.parse_ref.mod_path.split(os.path.sep)[-1].split(".")[0],
                source=self.parse_ref.source,
                doc=doc,
                body=body,
                is_imported=False,
                kid=kid if len(kid) else [ast.EmptyToken()],
            )
            return self.nu(mod)

        def element_with_doc(
            self, kid: list[ast.ElementStmt | ast.String]
        ) -> ast.ElementStmt:
            """Grammar rule.

            element_with_doc: doc_tag element
            """
            if isinstance(kid[1], ast.ElementStmt) and isinstance(kid[0], ast.String):
                kid[1].doc = kid[0]
                kid[1].add_kids_left([kid[0]])
                return self.nu(kid[1])
            else:
                raise self.ice()

        def element(self, kid: list[ast.AstNode]) -> ast.ElementStmt:
            """Grammar rule.

            element: py_code_block
                | include_stmt
                | import_stmt
                | ability
                | architype
                | free_code
                | test
                | global_var
            """
            if isinstance(kid[0], ast.ElementStmt):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def global_var(self, kid: list[ast.AstNode]) -> ast.GlobalVars:
            """Grammar rule.

            global_var: (KW_LET | KW_GLOBAL) access_tag? assignment_list SEMI
            """
            is_frozen = isinstance(kid[0], ast.Token) and kid[0].name == Tok.KW_LET
            access = kid[1] if isinstance(kid[1], ast.SubTag) else None
            assignments = kid[2] if access else kid[1]
            if isinstance(assignments, ast.SubNodeList):
                return self.nu(
                    ast.GlobalVars(
                        access=access,
                        assignments=assignments,
                        is_frozen=is_frozen,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def access_tag(self, kid: list[ast.AstNode]) -> ast.SubTag[ast.Token]:
            """Grammar rule.

            access_tag: COLON ( KW_PROT | KW_PUB | KW_PRIV )
            """
            if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.Token):
                return self.nu(
                    ast.SubTag[ast.Token](
                        tag=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def test(self, kid: list[ast.AstNode]) -> ast.Test:
            """Grammar rule.

            test: KW_TEST NAME? code_block
            """
            name = kid[1] if isinstance(kid[1], ast.Name) else kid[0]
            codeblock = kid[2] if name != kid[0] else kid[1]
            if isinstance(codeblock, ast.SubNodeList) and isinstance(
                name, (ast.Name, ast.Token)
            ):
                return self.nu(
                    ast.Test(
                        name=name,
                        body=codeblock,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def free_code(self, kid: list[ast.AstNode]) -> ast.ModuleCode:
            """Grammar rule.

            free_code: KW_WITH KW_ENTRY sub_name? code_block
            """
            name = kid[2] if isinstance(kid[2], ast.SubTag) else None
            codeblock = kid[3] if name else kid[2]
            if isinstance(codeblock, ast.SubNodeList):
                return self.nu(
                    ast.ModuleCode(
                        name=name,
                        body=codeblock,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def doc_tag(self, kid: list[ast.AstNode]) -> ast.String:
            """Grammar rule.

            doc_tag: ( STRING | DOC_STRING )
            """
            if isinstance(kid[0], ast.String):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def py_code_block(self, kid: list[ast.AstNode]) -> ast.PyInlineCode:
            """Grammar rule.

            py_code_block: PYNLINE
            """
            if isinstance(kid[0], ast.Token):
                return self.nu(
                    ast.PyInlineCode(
                        code=kid[0],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def import_stmt(self, kid: list[ast.AstNode]) -> ast.Import:
            """Grammar rule.

            import_stmt: KW_IMPORT sub_name KW_FROM import_path COMMA import_items SEMI
                    | KW_IMPORT sub_name import_path SEMI
            """
            lang = kid[1]
            path = kid[3] if isinstance(kid[3], ast.ModulePath) else kid[2]

            items = (
                kid[-2]
                if len(kid) > 4 and isinstance(kid[-2], ast.SubNodeList)
                else None
            )
            is_absorb = False
            if (
                isinstance(lang, ast.SubTag)
                and isinstance(path, ast.ModulePath)
                and (isinstance(items, ast.SubNodeList) or items is None)
            ):
                return self.nu(
                    ast.Import(
                        lang=lang,
                        path=path,
                        items=items,
                        is_absorb=is_absorb,
                        kid=kid,
                    )
                )

            else:
                raise self.ice()

        def include_stmt(self, kid: list[ast.AstNode]) -> ast.Import:
            """Grammar rule.

            include_stmt: KW_INCLUDE sub_name import_path SEMI
            """
            lang = kid[1]
            path = kid[3] if isinstance(kid[3], ast.ModulePath) else kid[2]
            is_absorb = True
            if isinstance(lang, ast.SubTag) and isinstance(path, ast.ModulePath):
                return self.nu(
                    ast.Import(
                        lang=lang,
                        path=path,
                        items=None,
                        is_absorb=is_absorb,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def import_path(self, kid: list[ast.AstNode]) -> ast.ModulePath:
            """Grammar rule.

            import_path: DOT? DOT? named_ref (DOT named_ref)* (KW_AS NAME)?
            """
            valid_path = [i for i in kid if isinstance(i, ast.Token)]
            alias = (
                kid[-1]
                if len(kid) > 2
                and isinstance(kid[-1], ast.Name)
                and isinstance(kid[-2], ast.Token)
                and kid[-2].name == Tok.KW_AS
                else None
            )
            if alias is not None:
                valid_path = valid_path[:-2]
            return self.nu(
                ast.ModulePath(
                    path=valid_path,
                    alias=alias,
                    kid=kid,
                )
            )

        def import_items(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.ModuleItem]:
            """Grammar rule.

            import_items: (import_item COMMA)* import_item
            """
            ret = ast.SubNodeList[ast.ModuleItem](
                items=[i for i in kid if isinstance(i, ast.ModuleItem)],
                kid=kid,
            )
            return self.nu(ret)

        def import_item(self, kid: list[ast.AstNode]) -> ast.ModuleItem:
            """Grammar rule.

            import_item: named_ref (KW_AS NAME)?
            """
            name = kid[0]
            alias = kid[2] if len(kid) > 1 else None
            if isinstance(name, ast.Name) and (
                alias is None or isinstance(alias, ast.Name)
            ):
                return self.nu(
                    ast.ModuleItem(
                        name=name,
                        alias=alias,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def architype(self, kid: list[ast.AstNode]) -> ast.ArchSpec:
            """Grammar rule.

            architype: decorators architype
                    | enum
                    | architype_def
                    | architype_decl
            """
            if isinstance(kid[0], ast.SubNodeList):
                if isinstance(kid[1], ast.ArchSpec):
                    kid[1].decorators = kid[0]
                    kid[1].add_kids_left([kid[0]])
                    return self.nu(kid[1])
                else:
                    raise self.ice()
            elif isinstance(kid[0], ast.ArchSpec):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def architype_decl(self, kid: list[ast.AstNode]) -> ast.ArchSpec:
            """Grammar rule.

            architype_decl: arch_type access_tag? NAME inherited_archs? (member_block | SEMI)
            """
            arch_type = kid[0]
            access = kid[1] if isinstance(kid[1], ast.SubTag) else None
            name = kid[2] if access else kid[1]
            inh = kid[-2] if isinstance(kid[-2], ast.SubNodeList) else None
            body = kid[-1] if isinstance(kid[-1], ast.SubNodeList) else None
            if isinstance(arch_type, ast.Token) and isinstance(name, ast.Name):
                return self.nu(
                    ast.Architype(
                        arch_type=arch_type,
                        name=name,
                        access=access,
                        base_classes=inh,
                        body=body,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def architype_def(self, kid: list[ast.AstNode]) -> ast.ArchDef:
            """Grammar rule.

            architype_def: abil_to_arch_chain member_block
            """
            if isinstance(kid[0], ast.ArchRefChain) and isinstance(
                kid[1], ast.SubNodeList
            ):
                return self.nu(
                    ast.ArchDef(
                        target=kid[0],
                        body=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def arch_type(self, kid: list[ast.AstNode]) -> ast.Token:
            """Grammar rule.

            arch_type: KW_WALKER
                    | KW_OBJECT
                    | KW_EDGE
                    | KW_NODE
            """
            if isinstance(kid[0], ast.Token):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def decorators(self, kid: list[ast.AstNode]) -> ast.SubNodeList[ast.Expr]:
            """Grammar rule.

            decorators: (DECOR_OP atomic_chain)+
            """
            valid_decors = [i for i in kid if isinstance(i, ast.Expr)]
            if len(valid_decors) == len(kid) / 2:
                return self.nu(
                    ast.SubNodeList[ast.Expr](
                        items=valid_decors,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def inherited_archs(self, kid: list[ast.AstNode]) -> ast.SubNodeList[ast.Expr]:
            """Grammar rule.

            inherited_archs: LT (atomic_chain COMMA)* atomic_chain GT
                           | COLON (atomic_chain COMMA)* atomic_chain COLON
            """
            valid_inh = [i for i in kid if isinstance(i, ast.Expr)]
            return self.nu(
                ast.SubNodeList[ast.Expr](
                    items=valid_inh,
                    kid=kid,
                )
            )

        def sub_name(self, kid: list[ast.AstNode]) -> ast.SubTag[ast.Name]:
            """Grammar rule.

            sub_name: COLON NAME
            """
            if isinstance(kid[1], ast.Name):
                return self.nu(
                    ast.SubTag[ast.Name](
                        tag=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def any_ref(self, kid: list[ast.AstNode]) -> ast.NameSpec:
            """Grammar rule.

            any_ref: named_ref
                    | arch_ref
            """
            if isinstance(kid[0], ast.NameSpec):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def named_ref(self, kid: list[ast.AstNode]) -> ast.NameSpec:
            """Grammar rule.

            named_ref: special_ref
                    | KWESC_NAME
                    | NAME
            """
            if isinstance(kid[0], ast.NameSpec):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def special_ref(self, kid: list[ast.AstNode]) -> ast.SpecialVarRef:
            """Grammar rule.

            special_ref: INIT_OP
                        | ROOT_OP
                        | SUPER_OP
                        | SELF_OP
                        | HERE_OP
            """
            if isinstance(kid[0], ast.Token):
                return self.nu(
                    ast.SpecialVarRef(
                        var=kid[0],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def enum(self, kid: list[ast.AstNode]) -> ast.Enum | ast.EnumDef:
            """Grammar rule.

            enum: enum_def
                | enum_decl
            """
            if isinstance(kid[0], (ast.Enum, ast.EnumDef)):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def enum_decl(self, kid: list[ast.AstNode]) -> ast.Enum:
            """Grammar rule.

            enum_decl: KW_ENUM access_tag? NAME inherited_archs? (enum_block | SEMI)
            """
            access = kid[1] if isinstance(kid[1], ast.SubTag) else None
            name = kid[2] if access else kid[1]
            inh = kid[-2] if isinstance(kid[-2], ast.SubNodeList) else None
            body = kid[-1] if isinstance(kid[-1], ast.SubNodeList) else None
            if isinstance(name, ast.Name):
                return self.nu(
                    ast.Enum(
                        name=name,
                        access=access,
                        base_classes=inh,
                        body=body,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def enum_def(self, kid: list[ast.AstNode]) -> ast.EnumDef:
            """Grammar rule.

            enum_def: arch_to_enum_chain enum_block
            """
            if isinstance(kid[0], ast.ArchRefChain) and isinstance(
                kid[1], ast.SubNodeList
            ):
                return self.nu(
                    ast.EnumDef(
                        target=kid[0],
                        body=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def enum_block(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.EnumBlockStmt]:
            """Grammar rule.

            enum_block: LBRACE ((enum_stmt COMMA)* enum_stmt)? RBRACE
            """
            ret = ast.SubNodeList[ast.EnumBlockStmt](
                items=[],
                kid=kid,
            )
            ret.items = [i for i in kid if isinstance(i, ast.EnumBlockStmt)]
            return self.nu(ret)

        def enum_stmt(self, kid: list[ast.AstNode]) -> ast.EnumBlockStmt:
            """Grammar rule.

            enum_stmt: NAME EQ expression
                    | NAME
                    | py_code_block
            """
            if isinstance(kid[0], ast.PyInlineCode):
                return self.nu(kid[0])
            if isinstance(kid[0], (ast.Name)):
                if len(kid) == 1:
                    kid[0].is_enum_singleton = True
                    return self.nu(kid[0])
                elif isinstance(kid[2], ast.Expr):
                    targ = ast.SubNodeList[ast.Expr](items=[kid[0]], kid=[kid[0]])
                    kid[0] = targ
                    return self.nu(
                        ast.Assignment(
                            target=targ,
                            value=kid[2],
                            type_tag=None,
                            kid=kid,
                        )
                    )
            raise self.ice()

        def ability(self, kid: list[ast.AstNode]) -> ast.Ability | ast.AbilityDef:
            """Grammar rule.

            ability: decorators? ability_def
                    | decorators? KW_ASYNC ability_decl
                    | decorators? ability_decl
            """
            if isinstance(kid[0], ast.SubNodeList):
                if isinstance(kid[1], (ast.Ability, ast.AbilityDef)):
                    for dec in kid[0].items:
                        if (
                            isinstance(dec, ast.NameSpec)
                            and dec.sym_name == "staticmethod"
                            and isinstance(kid[1], (ast.Ability))
                        ):
                            kid[1].is_static = True
                            kid[0].items.remove(dec)  # noqa: B038
                            break
                    if len(kid[0].items):
                        kid[1].decorators = kid[0]
                        kid[1].add_kids_left([kid[0]])
                    return self.nu(kid[1])
                else:
                    raise self.ice()
            elif isinstance(kid[0], (ast.Ability, ast.AbilityDef)):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def ability_decl(self, kid: list[ast.AstNode]) -> ast.Ability:
            """Grammar rule.

            ability_decl: KW_STATIC? KW_CAN access_tag? any_ref (func_decl | event_clause) (code_block | SEMI)
            """
            chomp = [*kid]
            is_static = (
                isinstance(chomp[0], ast.Token) and chomp[0].name == Tok.KW_STATIC
            )
            chomp = chomp[2:] if is_static else chomp[1:]
            access = chomp[0] if isinstance(chomp[0], ast.SubTag) else None
            chomp = chomp[1:] if access else chomp
            name = chomp[0]
            chomp = chomp[1:]
            is_func = isinstance(chomp[0], ast.FuncSignature)
            signature = chomp[0]
            chomp = chomp[1:]
            body = chomp[0] if isinstance(chomp[0], ast.SubNodeList) else None
            if isinstance(name, ast.NameSpec) and isinstance(
                signature, (ast.FuncSignature, ast.EventSignature)
            ):
                return self.nu(
                    ast.Ability(
                        name_ref=name,
                        is_func=is_func,
                        is_async=False,
                        is_static=is_static,
                        is_abstract=False,
                        access=access,
                        signature=signature,
                        body=body,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def ability_def(self, kid: list[ast.AstNode]) -> ast.AbilityDef:
            """Grammar rule.

            ability_def: arch_to_abil_chain (func_decl | event_clause) code_block
            """
            if (
                isinstance(kid[0], ast.ArchRefChain)
                and isinstance(kid[1], (ast.FuncSignature, ast.EventSignature))
                and isinstance(kid[2], ast.SubNodeList)
            ):
                return self.nu(
                    ast.AbilityDef(
                        target=kid[0],
                        signature=kid[1],
                        body=kid[2],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def abstract_ability(self, kid: list[ast.AstNode]) -> ast.Ability:
            """Grammar rule.

            abstract_ability: KW_STATIC? KW_CAN access_tag? any_ref (func_decl | event_clause) KW_ABSTRACT SEMI
            """
            chomp = [*kid]
            is_static = (
                isinstance(chomp[0], ast.Token) and chomp[0].name == Tok.KW_STATIC
            )
            chomp = chomp[1:] if is_static else chomp
            chomp = chomp[1:]
            access = chomp[0] if isinstance(chomp[0], ast.SubTag) else None
            chomp = chomp[1:] if access else chomp
            name = chomp[0]
            chomp = chomp[1:]
            is_func = isinstance(chomp[0], ast.FuncSignature)
            signature = chomp[0]
            chomp = chomp[1:]
            if isinstance(name, ast.NameSpec) and isinstance(
                signature, (ast.FuncSignature, ast.EventSignature)
            ):
                return self.nu(
                    ast.Ability(
                        name_ref=name,
                        is_func=is_func,
                        is_async=False,
                        is_static=is_static,
                        is_abstract=True,
                        access=access,
                        signature=signature,
                        body=None,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def event_clause(self, kid: list[ast.AstNode]) -> ast.EventSignature:
            """Grammar rule.

            event_clause: KW_WITH expression? (KW_EXIT | KW_ENTRY) return_type_tag?
            """
            type_specs = kid[1] if isinstance(kid[1], ast.Expr) else None
            return_spec = kid[-1] if isinstance(kid[-1], ast.SubTag) else None
            event = kid[2] if type_specs else kid[1]
            if isinstance(event, ast.Token) and (
                isinstance(return_spec, ast.SubTag) or return_spec is None
            ):
                return self.nu(
                    ast.EventSignature(
                        event=event,
                        arch_tag_info=type_specs,
                        return_type=return_spec,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def func_decl(self, kid: list[ast.AstNode]) -> ast.FuncSignature:
            """Grammar rule.

            func_decl: (LPAREN func_decl_params? RPAREN)? return_type_tag?
            """
            params = (
                kid[1] if len(kid) > 1 and isinstance(kid[1], ast.SubNodeList) else None
            )
            return_spec = (
                kid[-1] if len(kid) and isinstance(kid[-1], ast.SubTag) else None
            )
            if (isinstance(params, ast.SubNodeList) or params is None) and (
                isinstance(return_spec, ast.SubTag) or return_spec is None
            ):
                return self.nu(
                    ast.FuncSignature(
                        params=params,
                        return_type=return_spec,
                        kid=kid if len(kid) else [ast.EmptyToken()],
                    )
                )
            else:
                raise self.ice()

        def func_decl_params(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.ParamVar]:
            """Grammar rule.

            func_decl_params: (param_var COMMA)* param_var
            """
            ret = ast.SubNodeList[ast.ParamVar](
                items=[i for i in kid if isinstance(i, ast.ParamVar)],
                kid=kid,
            )
            return self.nu(ret)

        def param_var(self, kid: list[ast.AstNode]) -> ast.ParamVar:
            """Grammar rule.

            param_var: (STAR_POW | STAR_MUL)? NAME type_tag (EQ expression)?
            """
            star = (
                kid[0]
                if isinstance(kid[0], ast.Token) and kid[0].name != Tok.NAME
                else None
            )
            name = kid[1] if star else kid[0]
            type_tag = kid[2] if star else kid[1]
            value = kid[-1] if isinstance(kid[-1], ast.Expr) else None
            if isinstance(name, ast.Name) and isinstance(type_tag, ast.SubTag):
                return self.nu(
                    ast.ParamVar(
                        name=name,
                        type_tag=type_tag,
                        value=value,
                        unpack=star,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def member_block(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.ArchBlockStmt]:
            """Grammar rule.

            member_block: LBRACE member_stmt* RBRACE
            """
            ret = ast.SubNodeList[ast.ArchBlockStmt](
                items=[],
                kid=kid,
            )
            ret.items = [i for i in kid if isinstance(i, ast.ArchBlockStmt)]
            return self.nu(ret)

        def member_stmt(self, kid: list[ast.AstNode]) -> ast.ArchBlockStmt:
            """Grammar rule.

            member_stmt: doc_tag? py_code_block
                        | doc_tag? abstract_ability
                        | doc_tag? ability
                        | doc_tag? architype
                        | doc_tag? has_stmt
            """
            if isinstance(kid[0], ast.ArchBlockStmt):
                return self.nu(kid[0])
            elif (
                isinstance(kid[1], ast.ArchBlockStmt)
                and isinstance(kid[1], ast.AstDocNode)
                and isinstance(kid[0], ast.String)
            ):
                kid[1].doc = kid[0]
                kid[1].add_kids_left([kid[0]])
                return self.nu(kid[1])

            else:
                raise self.ice()

        def has_stmt(self, kid: list[ast.AstNode]) -> ast.ArchHas:
            """Grammar rule.

            has_stmt: KW_STATIC? (KW_LET | KW_HAS) access_tag? has_assign_list SEMI
            """
            chomp = [*kid]
            is_static = (
                isinstance(chomp[0], ast.Token) and chomp[0].name == Tok.KW_STATIC
            )
            chomp = chomp[1:] if is_static else chomp
            is_freeze = isinstance(chomp[0], ast.Token) and chomp[0].name == Tok.KW_LET
            chomp = chomp[1:]
            access = chomp[0] if isinstance(chomp[0], ast.SubTag) else None
            chomp = chomp[1:] if access else chomp
            assign = chomp[0]
            if isinstance(assign, ast.SubNodeList):
                return self.nu(
                    ast.ArchHas(
                        vars=assign,
                        is_static=is_static,
                        is_frozen=is_freeze,
                        access=access,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def has_assign_list(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.HasVar]:
            """Grammar rule.

            has_assign_list: (has_assign_list COMMA)? typed_has_clause
            """
            consume = None
            assign = None
            comma = None
            if isinstance(kid[0], ast.SubNodeList):
                consume = kid[0]
                comma = kid[1]
                assign = kid[2]
                new_kid = [*consume.kid, comma, assign]
            else:
                assign = kid[0]
                new_kid = [assign]
            valid_kid = [i for i in new_kid if isinstance(i, ast.HasVar)]
            return self.nu(
                ast.SubNodeList[ast.HasVar](
                    items=valid_kid,
                    kid=new_kid,
                )
            )

        def typed_has_clause(self, kid: list[ast.AstNode]) -> ast.HasVar:
            """Grammar rule.

            typed_has_clause: name_ref type_tag (EQ expression)?
            """
            name = kid[0]
            type_tag = kid[1]
            value = kid[-1] if isinstance(kid[-1], ast.Expr) else None
            if isinstance(name, ast.Name) and isinstance(type_tag, ast.SubTag):
                return self.nu(
                    ast.HasVar(
                        name=name,
                        type_tag=type_tag,
                        value=value,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def type_tag(self, kid: list[ast.AstNode]) -> ast.SubTag[ast.Expr]:
            """Grammar rule.

            type_tag: COLON expression
            """
            if isinstance(kid[1], ast.Expr):
                return self.nu(
                    ast.SubTag[ast.Expr](
                        tag=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def return_type_tag(self, kid: list[ast.AstNode]) -> ast.SubTag[ast.Expr]:
            """Grammar rule.

            return_type_tag: RETURN_HINT expression
            """
            if isinstance(kid[1], ast.Expr):
                return self.nu(
                    ast.SubTag[ast.Expr](
                        tag=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def builtin_type(self, kid: list[ast.AstNode]) -> ast.Token:
            """Grammar rule.

            builtin_type: TYP_TYPE
                        | TYP_ANY
                        | TYP_BOOL
                        | TYP_DICT
                        | TYP_SET
                        | TYP_TUPLE
                        | TYP_LIST
                        | TYP_FLOAT
                        | TYP_INT
                        | TYP_BYTES
                        | TYP_STRING
            """
            if isinstance(kid[0], ast.Token):
                return self.nu(
                    ast.BuiltinType(
                        name=kid[0].name,
                        file_path=self.parse_ref.mod_path,
                        value=kid[0].value,
                        line=kid[0].loc.first_line,
                        col_start=kid[0].loc.col_start,
                        col_end=kid[0].loc.col_end,
                        pos_start=kid[0].pos_start,
                        pos_end=kid[0].pos_end,
                        kid=kid[0].kid,
                    )
                )
            else:
                raise self.ice()

        def code_block(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.CodeBlockStmt]:
            """Grammar rule.

            code_block: LBRACE statement_list* RBRACE
            """
            if isinstance(kid[1], ast.SubNodeList):
                kid[1].add_kids_left([kid[0]])
                kid[1].add_kids_right([kid[2]])
                return self.nu(kid[1])
            else:
                return self.nu(
                    ast.SubNodeList[ast.CodeBlockStmt](
                        items=[],
                        kid=kid,
                    )
                )

        def statement_list(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.CodeBlockStmt]:
            """Grammar rule.

            statement_list: statement+
            """
            valid_stmt = [i for i in kid if isinstance(i, ast.CodeBlockStmt)]
            if len(valid_stmt) == len(kid):
                return self.nu(
                    ast.SubNodeList[ast.CodeBlockStmt](
                        items=valid_stmt,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def statement(self, kid: list[ast.AstNode]) -> ast.CodeBlockStmt:
            """Grammar rule.

            statement: py_code_block
                    | walker_stmt
                    | return_stmt SEMI
                    | report_stmt SEMI
                    | delete_stmt SEMI
                    | ctrl_stmt SEMI
                    | assert_stmt SEMI
                    | raise_stmt SEMI
                    | with_stmt
                    | while_stmt
                    | for_stmt
                    | try_stmt
                    | if_stmt
                    | expression SEMI
                    | yield_expr SEMI
                    | static_assignment
                    | assignment SEMI
                    | global_ref SEMI
                    | nonlocal_ref SEMI
                    | typed_ctx_block
                    | ability
                    | architype
                    | import_stmt
                    | SEMI
            """
            if isinstance(kid[0], ast.CodeBlockStmt) and len(kid) < 2:
                return self.nu(kid[0])
            elif isinstance(kid[0], ast.Expr):
                return ast.ExprStmt(
                    expr=kid[0],
                    in_fstring=False,
                    kid=kid,
                )
            elif isinstance(kid[0], ast.CodeBlockStmt):
                kid[0].add_kids_right([kid[1]])
                return self.nu(kid[0])
            else:
                raise self.ice()

        def typed_ctx_block(self, kid: list[ast.AstNode]) -> ast.TypedCtxBlock:
            """Grammar rule.

            typed_ctx_block: RETURN_HINT expression code_block
            """
            if isinstance(kid[1], ast.Expr) and isinstance(kid[2], ast.SubNodeList):
                return self.nu(
                    ast.TypedCtxBlock(
                        type_ctx=kid[1],
                        body=kid[2],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def if_stmt(self, kid: list[ast.AstNode]) -> ast.IfStmt:
            """Grammar rule.

            if_stmt: KW_IF expression code_block (elif_stmt | else_stmt)?
            """
            if isinstance(kid[1], ast.Expr) and isinstance(kid[2], ast.SubNodeList):
                return self.nu(
                    ast.IfStmt(
                        condition=kid[1],
                        body=kid[2],
                        else_body=kid[3]
                        if len(kid) > 3
                        and isinstance(kid[3], (ast.ElseStmt, ast.ElseIf))
                        else None,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def elif_stmt(self, kid: list[ast.AstNode]) -> ast.ElseIf:
            """Grammar rule.

            elif_stmt: KW_ELIF expression code_block (elif_stmt | else_stmt)?
            """
            if isinstance(kid[1], ast.Expr) and isinstance(kid[2], ast.SubNodeList):
                return self.nu(
                    ast.ElseIf(
                        condition=kid[1],
                        body=kid[2],
                        else_body=kid[3]
                        if len(kid) > 3
                        and isinstance(kid[3], (ast.ElseStmt, ast.ElseIf))
                        else None,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def else_stmt(self, kid: list[ast.AstNode]) -> ast.ElseStmt:
            """Grammar rule.

            else_stmt: KW_ELSE code_block
            """
            if isinstance(kid[1], ast.SubNodeList):
                return self.nu(
                    ast.ElseStmt(
                        body=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def try_stmt(self, kid: list[ast.AstNode]) -> ast.TryStmt:
            """Grammar rule.

            try_stmt: KW_TRY code_block except_list? else_stmt? finally_stmt?
            """
            chomp = [*kid][1:]
            block = chomp[0]
            chomp = chomp[1:]
            except_list = (
                chomp[0]
                if len(chomp) and isinstance(chomp[0], ast.SubNodeList)
                else None
            )
            chomp = chomp[1:] if except_list else chomp
            else_stmt = (
                chomp[0] if len(chomp) and isinstance(chomp[0], ast.ElseStmt) else None
            )
            chomp = chomp[1:] if else_stmt else chomp
            finally_stmt = (
                chomp[0]
                if len(chomp) and isinstance(chomp[0], ast.FinallyStmt)
                else None
            )
            if isinstance(block, ast.SubNodeList):
                return self.nu(
                    ast.TryStmt(
                        body=block,
                        excepts=except_list,
                        else_body=else_stmt,
                        finally_body=finally_stmt,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def except_list(self, kid: list[ast.AstNode]) -> ast.SubNodeList[ast.Except]:
            """Grammar rule.

            except_list: except_def+
            """
            valid_kid = [i for i in kid if isinstance(i, ast.Except)]
            if len(valid_kid) == len(kid):
                return self.nu(
                    ast.SubNodeList[ast.Except](
                        items=valid_kid,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def except_def(self, kid: list[ast.AstNode]) -> ast.Except:
            """Grammar rule.

            except_def: KW_EXCEPT expression (KW_AS NAME)? code_block
            """
            ex_type = kid[1]
            name = kid[3] if len(kid) > 3 and isinstance(kid[3], ast.Name) else None
            body = kid[-1]
            if isinstance(ex_type, ast.Expr) and isinstance(body, ast.SubNodeList):
                return self.nu(
                    ast.Except(
                        ex_type=ex_type,
                        name=name,
                        body=body,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def finally_stmt(self, kid: list[ast.AstNode]) -> ast.FinallyStmt:
            """Grammar rule.

            finally_stmt: KW_FINALLY code_block
            """
            if isinstance(kid[1], ast.SubNodeList):
                return self.nu(
                    ast.FinallyStmt(
                        body=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def for_stmt(self, kid: list[ast.AstNode]) -> ast.IterForStmt | ast.InForStmt:
            """Grammar rule.

            for_stmt: KW_ASYNC? KW_FOR assignment KW_TO expression KW_BY
                        expression code_block else_stmt?
                    | KW_ASYNC? KW_FOR expression KW_IN expression code_block else_stmt?
            """
            chomp = [*kid]
            is_async = bool(
                isinstance(chomp[0], ast.Token) and chomp[0].name == Tok.KW_ASYNC
            )
            chomp = chomp[1:] if is_async else chomp
            if isinstance(chomp[1], ast.Assignment):
                if (
                    isinstance(chomp[3], ast.Expr)
                    and isinstance(chomp[5], ast.Assignment)
                    and isinstance(chomp[6], ast.SubNodeList)
                ):
                    return self.nu(
                        ast.IterForStmt(
                            is_async=is_async,
                            iter=chomp[1],
                            condition=chomp[3],
                            count_by=chomp[5],
                            body=chomp[6],
                            else_body=chomp[-1]
                            if isinstance(chomp[-1], ast.ElseStmt)
                            else None,
                            kid=kid,
                        )
                    )
                else:
                    raise self.ice()
            elif isinstance(chomp[1], ast.Expr):
                if isinstance(chomp[3], ast.Expr) and isinstance(
                    chomp[4], ast.SubNodeList
                ):
                    return self.nu(
                        ast.InForStmt(
                            is_async=is_async,
                            target=chomp[1],
                            collection=chomp[3],
                            body=chomp[4],
                            else_body=chomp[-1]
                            if isinstance(chomp[-1], ast.ElseStmt)
                            else None,
                            kid=kid,
                        )
                    )
                else:
                    raise self.ice()
            else:
                raise self.ice()

        def while_stmt(self, kid: list[ast.AstNode]) -> ast.WhileStmt:
            """Grammar rule.

            while_stmt: KW_WHILE expression code_block
            """
            if isinstance(kid[1], ast.Expr) and isinstance(kid[2], ast.SubNodeList):
                return self.nu(
                    ast.WhileStmt(
                        condition=kid[1],
                        body=kid[2],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def with_stmt(self, kid: list[ast.AstNode]) -> ast.WithStmt:
            """Grammar rule.

            with_stmt: KW_ASYNC? KW_WITH expr_as_list code_block
            """
            chomp = [*kid]
            is_async = bool(
                isinstance(chomp[0], ast.Token) and chomp[0].name == Tok.KW_ASYNC
            )
            chomp = chomp[1:] if is_async else chomp
            if isinstance(chomp[1], ast.SubNodeList) and isinstance(
                chomp[2], ast.SubNodeList
            ):
                return self.nu(
                    ast.WithStmt(
                        is_async=is_async,
                        exprs=chomp[1],
                        body=chomp[2],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def expr_as_list(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.ExprAsItem]:
            """Grammar rule.

            expr_as_list: (expr_as COMMA)* expr_as
            """
            ret = ast.SubNodeList[ast.ExprAsItem](
                items=[i for i in kid if isinstance(i, ast.ExprAsItem)],
                kid=kid,
            )
            return self.nu(ret)

        def expr_as(self, kid: list[ast.AstNode]) -> ast.ExprAsItem:
            """Grammar rule.

            expr_as: expression (KW_AS expression)?
            """
            expr = kid[0]
            alias = kid[2] if len(kid) > 1 else None
            if isinstance(expr, ast.Expr) and (
                alias is None or isinstance(alias, ast.Expr)
            ):
                return self.nu(
                    ast.ExprAsItem(
                        expr=expr,
                        alias=alias,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def raise_stmt(self, kid: list[ast.AstNode]) -> ast.RaiseStmt:
            """Grammar rule.

            raise_stmt: KW_RAISE (expression (KW_FROM expression)?)?
            """
            chomp = [*kid][1:]
            e_type = (
                chomp[0] if len(chomp) > 0 and isinstance(chomp[0], ast.Expr) else None
            )
            chomp = chomp[2:] if e_type and len(chomp) > 1 else chomp[1:]
            e = chomp[0] if len(chomp) > 0 and isinstance(chomp[0], ast.Expr) else None
            return self.nu(
                ast.RaiseStmt(
                    cause=e_type,
                    from_target=e,
                    kid=kid,
                )
            )

        def assert_stmt(self, kid: list[ast.AstNode]) -> ast.AssertStmt:
            """Grammar rule.

            assert_stmt: KW_ASSERT expression (COMMA expression)?
            """
            condition = kid[1]
            error_msg = kid[3] if len(kid) > 3 else None
            if isinstance(condition, ast.Expr):
                return self.nu(
                    ast.AssertStmt(
                        condition=condition,
                        error_msg=error_msg
                        if isinstance(error_msg, ast.Expr)
                        else None,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def ctrl_stmt(self, kid: list[ast.AstNode]) -> ast.CtrlStmt:
            """Grammar rule.

            ctrl_stmt: KW_SKIP | KW_BREAK | KW_CONTINUE
            """
            if isinstance(kid[0], ast.Token):
                return self.nu(
                    ast.CtrlStmt(
                        ctrl=kid[0],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def delete_stmt(self, kid: list[ast.AstNode]) -> ast.DeleteStmt:
            """Grammar rule.

            delete_stmt: KW_DELETE expression
            """
            if isinstance(kid[1], ast.Expr):
                return self.nu(
                    ast.DeleteStmt(
                        target=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def report_stmt(self, kid: list[ast.AstNode]) -> ast.ReportStmt:
            """Grammar rule.

            report_stmt: KW_REPORT expression
            """
            if isinstance(kid[1], ast.Expr):
                return self.nu(
                    ast.ReportStmt(
                        expr=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def return_stmt(self, kid: list[ast.AstNode]) -> ast.ReturnStmt:
            """Grammar rule.

            return_stmt: KW_RETURN expression?
            """
            if len(kid) > 1:
                return self.nu(
                    ast.ReturnStmt(
                        expr=kid[1] if isinstance(kid[1], ast.Expr) else None,
                        kid=kid,
                    )
                )
            else:
                return self.nu(
                    ast.ReturnStmt(
                        expr=None,
                        kid=kid,
                    )
                )

        def walker_stmt(self, kid: list[ast.AstNode]) -> ast.CodeBlockStmt:
            """Grammar rule.

            walker_stmt: disengage_stmt | revisit_stmt | visit_stmt | ignore_stmt
            """
            if isinstance(kid[0], ast.CodeBlockStmt):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def ignore_stmt(self, kid: list[ast.AstNode]) -> ast.IgnoreStmt:
            """Grammar rule.

            ignore_stmt: KW_IGNORE expression SEMI
            """
            if isinstance(kid[1], ast.Expr):
                return self.nu(
                    ast.IgnoreStmt(
                        target=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def visit_stmt(self, kid: list[ast.AstNode]) -> ast.VisitStmt:
            """Grammar rule.

            visit_stmt: KW_VISIT (inherited_archs)? expression (else_stmt | SEMI)
            """
            sub_name = kid[1] if isinstance(kid[1], ast.SubNodeList) else None
            target = kid[2] if sub_name else kid[1]
            else_body = kid[-1] if isinstance(kid[-1], ast.ElseStmt) else None
            if isinstance(target, ast.Expr):
                return self.nu(
                    ast.VisitStmt(
                        vis_type=sub_name,
                        target=target,
                        else_body=else_body,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def revisit_stmt(self, kid: list[ast.AstNode]) -> ast.RevisitStmt:
            """Grammar rule.

            revisit_stmt: KW_REVISIT expression? (else_stmt | SEMI)
            """
            target = kid[1] if isinstance(kid[1], ast.Expr) else None
            else_body = kid[-1] if isinstance(kid[-1], ast.ElseStmt) else None
            return self.nu(
                ast.RevisitStmt(
                    hops=target,
                    else_body=else_body,
                    kid=kid,
                )
            )

        def disengage_stmt(self, kid: list[ast.AstNode]) -> ast.DisengageStmt:
            """Grammar rule.

            disengage_stmt: KW_DISENGAGE SEMI
            """
            return self.nu(
                ast.DisengageStmt(
                    kid=kid,
                )
            )

        def global_ref(self, kid: list[ast.AstNode]) -> ast.GlobalStmt:
            """Grammar rule.

            global_ref: GLOBAL_OP name_list
            """
            if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.SubNodeList):
                return self.nu(ast.GlobalStmt(target=kid[1], kid=kid))
            else:
                raise self.ice()

        def nonlocal_ref(self, kid: list[ast.AstNode]) -> ast.NonLocalStmt:
            """Grammar rule.

            nonlocal_ref: NONLOCAL_OP name_list
            """
            if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.SubNodeList):
                return self.nu(ast.NonLocalStmt(target=kid[1], kid=kid))
            else:
                raise self.ice()

        def assignment(self, kid: list[ast.AstNode]) -> ast.Assignment:
            """Grammar rule.

            assignment: KW_LET? (atomic_chain EQ)+ (yield_stmt | expression)
                    | atomic_chain type_tag (EQ (yield_stmt | expression))?
                    | atomic_chain aug_op (yield_stmt | expression)
            """
            chomp = [*kid]
            is_frozen = isinstance(chomp[0], ast.Token) and chomp[0].name == Tok.KW_LET
            is_aug = None
            assignees = []
            chomp = chomp[1:] if is_frozen else chomp
            if (
                len(chomp) > 1
                and isinstance(chomp[1], ast.Token)
                and chomp[1].name != Tok.EQ
            ):
                assignees += [chomp[0]]
                is_aug = chomp[1]
                chomp = chomp[2:]
            elif (
                len(chomp) > 1
                and isinstance(chomp[1], ast.Token)
                and chomp[1].name == Tok.EQ
            ):
                while (
                    isinstance(chomp[0], ast.Expr)
                    and len(chomp) > 1
                    and isinstance(chomp[1], ast.Token)
                    and chomp[1].name == Tok.EQ
                ):
                    assignees += [chomp[0], chomp[1]]
                    chomp = chomp[2:]
            elif isinstance(chomp[0], ast.Expr):
                assignees += [chomp[0]]
                chomp = chomp[1:]
            else:
                raise self.ice()
            valid_assignees = [i for i in assignees if isinstance(i, (ast.Expr))]
            new_targ = ast.SubNodeList[ast.Expr](
                items=valid_assignees,
                kid=assignees,
            )
            kid = [x for x in kid if x not in assignees]
            kid.insert(1, new_targ) if is_frozen else kid.insert(0, new_targ)
            type_tag = (
                chomp[0]
                if len(chomp) > 0 and isinstance(chomp[0], ast.SubTag)
                else None
            )
            chomp = chomp[1:] if type_tag else chomp
            if (
                len(chomp) > 0
                and isinstance(chomp[0], ast.Token)
                and chomp[0].name == Tok.EQ
            ):
                chomp = chomp[1:]
            value = (
                chomp[0]
                if len(chomp) > 0 and isinstance(chomp[0], (ast.YieldExpr, ast.Expr))
                else None
            )
            if is_aug:
                return self.nu(
                    ast.Assignment(
                        target=new_targ,
                        type_tag=type_tag,
                        value=value,
                        mutable=is_frozen,
                        aug_op=is_aug,
                        kid=kid,
                    )
                )
            return self.nu(
                ast.Assignment(
                    target=new_targ,
                    type_tag=type_tag,
                    value=value,
                    mutable=is_frozen,
                    kid=kid,
                )
            )

        def expression(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            expression: pipe KW_IF expression KW_ELSE expression
                    | pipe
                    | lamda_expr
            """
            if len(kid) > 1:
                if (
                    isinstance(kid[0], ast.Expr)
                    and isinstance(kid[2], ast.Expr)
                    and isinstance(kid[4], ast.Expr)
                ):
                    return self.nu(
                        ast.IfElseExpr(
                            value=kid[0],
                            condition=kid[2],
                            else_value=kid[4],
                            kid=kid,
                        )
                    )
                else:
                    raise self.ice()
            elif isinstance(kid[0], ast.Expr):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def binary_expr_unwind(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Binary expression helper."""
            if len(kid) > 1:
                if (
                    isinstance(kid[0], ast.Expr)
                    and isinstance(kid[1], (ast.Token, ast.DisconnectOp, ast.ConnectOp))
                    and isinstance(kid[2], ast.Expr)
                ):
                    return self.nu(
                        ast.BinaryExpr(
                            left=kid[0],
                            op=kid[1],
                            right=kid[2],
                            kid=kid,
                        )
                    )
                else:
                    raise self.ice()
            elif isinstance(kid[0], ast.Expr):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def lambda_expr(self, kid: list[ast.AstNode]) -> ast.LambdaExpr:
            """Grammar rule.

            lamda_expr: KW_WITH func_decl_params? return_type_tag? KW_CAN expression
            """
            chomp = [*kid][1:]
            params = chomp[0] if isinstance(chomp[0], ast.SubNodeList) else None
            chomp = chomp[1:] if params else chomp
            return_type = chomp[0] if isinstance(chomp[0], ast.SubTag) else None
            chomp = chomp[1:] if return_type else chomp
            chomp = chomp[1:]
            sig_kid: list[ast.AstNode] = []
            if params:
                sig_kid.append(params)
            if return_type:
                sig_kid.append(return_type)
            signature = ast.FuncSignature(
                params=params,
                return_type=return_type,
                kid=sig_kid,
            )
            new_kid = [i for i in kid if i != params and i != return_type]
            new_kid.insert(1, signature)
            if isinstance(chomp[0], ast.Expr):
                return self.nu(
                    ast.LambdaExpr(
                        signature=signature,
                        body=chomp[0],
                        kid=new_kid,
                    )
                )
            else:
                raise self.ice()

        def pipe(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            pipe: pipe_back PIPE_FWD pipe
                | pipe_back
            """
            return self.binary_expr_unwind(kid)

        def pipe_back(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            pipe_back: elvis_check PIPE_BKWD pipe_back
                     | elvis_check
            """
            return self.binary_expr_unwind(kid)

        def elvis_check(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            elvis_check: bitwise_or ELVIS_OP elvis_check
                       | bitwise_or
            """
            return self.binary_expr_unwind(kid)

        def bitwise_or(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            bitwise_or: bitwise_xor BW_OR bitwise_or
                      | bitwise_xor
            """
            return self.binary_expr_unwind(kid)

        def bitwise_xor(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            bitwise_xor: bitwise_and BW_XOR bitwise_xor
                       | bitwise_and
            """
            return self.binary_expr_unwind(kid)

        def bitwise_and(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            bitwise_and: shift BW_AND bitwise_and
                       | shift
            """
            return self.binary_expr_unwind(kid)

        def shift(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            shift: logical RSHIFT shift
                 | logical LSHIFT shift
                 | logical
            """
            return self.binary_expr_unwind(kid)

        def logical(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            logical: NOT logical
                   | compare KW_OR logical
                   | compare KW_AND logical
                   | compare
            """
            if len(kid) == 2:
                if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.Expr):
                    return self.nu(
                        ast.UnaryExpr(
                            op=kid[0],
                            operand=kid[1],
                            kid=kid,
                        )
                    )
                else:
                    raise self.ice()
            return self.binary_expr_unwind(kid)

        def compare(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            compare: arithmetic cmp_op compare
                   | arithmetic
            """
            return self.binary_expr_unwind(kid)

        def arithmetic(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            arithmetic: term MINUS arithmetic
                      | term PLUS arithmetic
                      | term
            """
            return self.binary_expr_unwind(kid)

        def term(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            term: factor MOD term
                 | factor DIV term
                 | factor FLOOR_DIV term
                 | factor STAR_MUL term
                 | factor
            """
            return self.binary_expr_unwind(kid)

        def factor(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            factor: power
                  | BW_NOT factor
                  | MINUS factor
                  | PLUS factor
            """
            if len(kid) == 2:
                if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.Expr):
                    return self.nu(
                        ast.UnaryExpr(
                            op=kid[0],
                            operand=kid[1],
                            kid=kid,
                        )
                    )
                else:
                    raise self.ice()
            return self.binary_expr_unwind(kid)

        def power(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            power: connect STAR_POW power
                  | connect
            """
            return self.binary_expr_unwind(kid)

        def connect(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            connect: atomic_pipe
                   | atomic_pipe connect_op connect
                   | atomic_pipe disconnect_op connect
            """
            return self.binary_expr_unwind(kid)

        def atomic_pipe(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            atomic_pipe: atomic_pipe_back
                       | atomic_pipe KW_SPAWN atomic_pipe_back
                       | atomic_pipe A_PIPE_FWD atomic_pipe_back
            """
            return self.binary_expr_unwind(kid)

        def atomic_pipe_back(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            atomic_pipe_back: (atomic_pipe_back A_PIPE_BKWD)? ds_spawn
            """
            return self.binary_expr_unwind(kid)

        def ds_spawn(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            ds_spawn: (ds_spawn KW_SPAWN)? unpack
            """
            return self.binary_expr_unwind(kid)

        def unpack(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            unpack: ref
                | STAR_MUL unpack
                | STAR_POW unpack
            """
            if len(kid) == 2:
                if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.Expr):
                    return self.nu(
                        ast.UnaryExpr(
                            op=kid[0],
                            operand=kid[1],
                            kid=kid,
                        )
                    )
                else:
                    raise self.ice()
            return self.binary_expr_unwind(kid)

        def ref(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            ref: walrus_assign
               | BW_AND walrus_assign
            """
            if len(kid) == 2:
                if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.Expr):
                    return self.nu(
                        ast.UnaryExpr(
                            op=kid[0],
                            operand=kid[1],
                            kid=kid,
                        )
                    )
                else:
                    raise self.ice()
            return self.binary_expr_unwind(kid)

        def walrus_assign(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            walrus_assign: pipe_call walrus_op walrus_assign
                         | pipe_call
            """
            return self.binary_expr_unwind(kid)

        def pipe_call(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            pipe_call: atomic_chain
                | PIPE_FWD atomic_chain
                | A_PIPE_FWD atomic_chain
                | KW_SPAWN atomic_chain
                | KW_AWAIT atomic_chain
            """
            if len(kid) == 2:
                if (
                    isinstance(kid[0], ast.Token)
                    and kid[0].name == Tok.KW_AWAIT
                    and isinstance(kid[1], ast.Expr)
                ):
                    return self.nu(
                        ast.AwaitExpr(
                            target=kid[1],
                            kid=kid,
                        )
                    )
                elif isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.Expr):
                    return self.nu(
                        ast.UnaryExpr(
                            op=kid[0],
                            operand=kid[1],
                            kid=kid,
                        )
                    )
                else:
                    raise self.ice()
            return self.binary_expr_unwind(kid)

        def aug_op(self, kid: list[ast.AstNode]) -> ast.Token:
            """Grammar rule.

            aug_op: RSHIFT_EQ
                     | LSHIFT_EQ
                     | BW_NOT_EQ
                     | BW_XOR_EQ
                     | BW_OR_EQ
                     | BW_AND_EQ
                     | MOD_EQ
                     | DIV_EQ
                     | FLOOR_DIV_EQ
                     | MUL_EQ
                     | SUB_EQ
                     | ADD_EQ
                     | WALRUS_EQ
            """
            if isinstance(kid[0], ast.Token):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def cmp_op(self, kid: list[ast.AstNode]) -> ast.Token:
            """Grammar rule.

            cmp_op: KW_ISN
                  | KW_IS
                  | KW_NIN
                  | KW_IN
                  | NE
                  | GTE
                  | LTE
                  | GT
                  | LT
                  | EE
            """
            if isinstance(kid[0], ast.Token):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def atomic_chain(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            atomic_chain: atomic_chain NULL_OK? (filter_compr | assign_compr | edge_op_ref | index_slice)
                        | atomic_chain NULL_OK? (DOT_BKWD | DOT_FWD | DOT) any_ref
                        | (atomic_call | atom)
            """
            if len(kid) < 2 and isinstance(kid[0], ast.Expr):
                return self.nu(kid[0])
            chomp = [*kid]
            target = chomp[0]
            chomp = chomp[1:]
            is_null_ok = False
            if isinstance(chomp[0], ast.Token) and chomp[0].name == Tok.NULL_OK:
                is_null_ok = True
                chomp = chomp[1:]
            if (
                len(chomp) == 1
                and isinstance(chomp[0], ast.AtomExpr)
                and isinstance(target, ast.Expr)
            ):
                return self.nu(
                    ast.AtomTrailer(
                        target=target,
                        right=chomp[0],
                        is_null_ok=is_null_ok,
                        is_attr=False,
                        kid=kid,
                    )
                )
            elif (
                len(chomp) > 1
                and isinstance(chomp[0], ast.Token)
                and isinstance(chomp[1], ast.AtomExpr)
                and isinstance(target, ast.Expr)
            ):
                return self.nu(
                    ast.AtomTrailer(
                        target=target if chomp[0].name != Tok.DOT_BKWD else chomp[1],
                        right=chomp[1] if chomp[0].name != Tok.DOT_BKWD else target,
                        is_null_ok=is_null_ok,
                        is_attr=True,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def atomic_call(self, kid: list[ast.AstNode]) -> ast.FuncCall:
            """Grammar rule.

            atomic_call: atomic_chain LPAREN param_list? RPAREN
            """
            if (
                len(kid) == 4
                and isinstance(kid[0], ast.Expr)
                and isinstance(kid[2], ast.SubNodeList)
            ):
                return self.nu(ast.FuncCall(target=kid[0], params=kid[2], kid=kid))
            elif len(kid) == 3 and isinstance(kid[0], ast.Expr):
                return self.nu(ast.FuncCall(target=kid[0], params=None, kid=kid))
            else:
                raise self.ice()

        def index_slice(self, kid: list[ast.AstNode]) -> ast.IndexSlice | ast.ListVal:
            """Grammar rule.

            index_slice: LSQUARE expression? COLON expression? (COLON expression?)? RSQUARE
                       | list_val
            """
            if len(kid) == 1 and isinstance(kid[0], ast.ListVal):
                expr = None
                if not kid[0].values or len(kid[0].values.items) < 1:
                    self.parse_ref.error("Empty list slice not allowed", kid[0].values)
                elif len(kid[0].values.items) == 1:
                    expr = kid[0].values.items[0]  # TODO: Loses braces
                else:
                    expr = ast.TupleVal(values=kid[0].values, kid=kid[0].kid)
                if expr is None:
                    raise self.ice()
                return self.nu(
                    ast.IndexSlice(
                        start=expr, stop=None, step=None, is_range=False, kid=[expr]
                    )
                )
            chomp = [*kid]
            chomp = chomp[1:]
            expr1 = chomp[0] if isinstance(chomp[0], ast.Expr) else None
            expr2 = (
                chomp[1]
                if isinstance(chomp[0], ast.Token)
                and chomp[0].name == Tok.COLON
                and isinstance(chomp[1], ast.Expr)
                else None
            )
            chomp = chomp[1:]
            expr2 = (
                chomp[1]
                if isinstance(chomp[0], ast.Token)
                and chomp[0].name == Tok.COLON
                and len(chomp) > 1
                and isinstance(chomp[1], ast.Expr)
                else expr2
            )
            expr3 = None
            if len(chomp) > 1:
                chomp = chomp[1:]
                expr3 = (
                    chomp[1]
                    if isinstance(chomp[0], ast.Token)
                    and chomp[0].name == Tok.COLON
                    and isinstance(chomp[1], ast.Expr)
                    else None
                )
                if len(chomp) > 1:
                    chomp = chomp[1:]
                    expr3 = (
                        chomp[1]
                        if isinstance(chomp[0], ast.Token)
                        and chomp[0].name == Tok.COLON
                        and len(chomp) > 1
                        and isinstance(chomp[1], ast.Expr)
                        else expr3
                    )
            return self.nu(
                ast.IndexSlice(
                    start=expr1,
                    stop=expr2,
                    step=expr3,
                    is_range=True,
                    kid=kid,
                )
            )

        def atom(self, kid: list[ast.AstNode]) -> ast.Expr:
            """Grammar rule.

            atom: edge_op_ref
                 | any_ref
                 | LPAREN (expression | yield_expr) RPAREN
                 | atom_collection
                 | atom_literal
            """
            if len(kid) == 1:
                if isinstance(kid[0], ast.AtomExpr):
                    return self.nu(kid[0])
                else:
                    raise self.ice()
            elif len(kid) == 3:
                if (
                    isinstance(kid[0], ast.Token)
                    and isinstance(kid[1], (ast.Expr, ast.YieldExpr))
                    and isinstance(kid[2], ast.Token)
                ):
                    ret = ast.AtomUnit(value=kid[1], is_paren=True, kid=kid)
                    # ret.add_kids_left([kid[0]])
                    # ret.add_kids_right([kid[2]])
                    return self.nu(ret)
                else:
                    raise self.ice()
            else:
                raise self.ice()

        def yield_expr(self, kid: list[ast.AstNode]) -> ast.YieldExpr:
            """Grammar rule.

            yield_expr:
                | KW_YIELD KW_FROM expression
                | KW_YIELD expression
            """
            if isinstance(kid[-1], ast.Expr):
                return self.nu(
                    ast.YieldExpr(
                        expr=kid[-1],
                        with_from=len(kid) > 2,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def atom_literal(self, kid: list[ast.AstNode]) -> ast.AtomExpr:
            """Grammar rule.

            atom_literal: builtin_type
                        | NULL
                        | BOOL
                        | multistring
                        | FLOAT
                        | OCT
                        | BIN
                        | HEX
                        | INT
            """
            if isinstance(kid[0], ast.AtomExpr):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def atom_collection(self, kid: list[ast.AstNode]) -> ast.AtomExpr:
            """Grammar rule.

            atom_collection: dict_compr
                           | set_compr
                           | gen_compr
                           | list_compr
                           | dict_val
                           | set_val
                           | tuple_val
                           | list_val
            """
            if isinstance(kid[0], ast.AtomExpr):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def multistring(self, kid: list[ast.AstNode]) -> ast.AtomExpr:
            """Grammar rule.

            multistring: (fstring | STRING)+
            """
            valid_strs = [i for i in kid if isinstance(i, (ast.String, ast.FString))]
            if len(valid_strs) == len(kid):
                return self.nu(
                    ast.MultiString(
                        strings=valid_strs,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def fstring(self, kid: list[ast.AstNode]) -> ast.FString:
            """Grammar rule.

            fstring: FSTR_START fstr_parts FSTR_END
            """
            if len(kid) == 2:
                return self.nu(
                    ast.FString(
                        parts=None,
                        kid=kid,
                    )
                )
            elif isinstance(kid[1], ast.SubNodeList):
                return self.nu(
                    ast.FString(
                        parts=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def fstr_parts(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.String | ast.ExprStmt]:
            """Grammar rule.

            fstr_parts: (FSTR_PIECE | FSTR_BESC | LBRACE expression RBRACE | fstring)*
            """
            valid_parts: list[ast.String | ast.ExprStmt] = [
                i
                if isinstance(i, ast.String)
                else ast.ExprStmt(expr=i, in_fstring=True, kid=[i])
                for i in kid
                if isinstance(i, ast.Expr)
            ]
            return self.nu(
                ast.SubNodeList[ast.String | ast.ExprStmt](
                    items=valid_parts,
                    kid=valid_parts,
                )
            )

        def list_val(self, kid: list[ast.AstNode]) -> ast.ListVal:
            """Grammar rule.

            list_val: LSQUARE expr_list? RSQUARE
            """
            if len(kid) == 2:
                return self.nu(
                    ast.ListVal(
                        values=None,
                        kid=kid,
                    )
                )
            elif isinstance(kid[1], ast.SubNodeList):
                return self.nu(
                    ast.ListVal(
                        values=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def tuple_val(self, kid: list[ast.AstNode]) -> ast.TupleVal:
            """Grammar rule.

            tuple_val: LPAREN tuple_list? RPAREN
            """
            if len(kid) == 2:
                return self.nu(
                    ast.TupleVal(
                        values=None,
                        kid=kid,
                    )
                )
            elif isinstance(kid[1], ast.SubNodeList):
                return self.nu(
                    ast.TupleVal(
                        values=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def set_val(self, kid: list[ast.AstNode]) -> ast.SetVal:
            """Grammar rule.

            set_val: LBRACE expr_list RBRACE
            """
            if len(kid) == 2:
                return self.nu(
                    ast.SetVal(
                        values=None,
                        kid=kid,
                    )
                )
            elif isinstance(kid[1], ast.SubNodeList):
                return self.nu(
                    ast.SetVal(
                        values=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def expr_list(self, kid: list[ast.AstNode]) -> ast.SubNodeList[ast.Expr]:
            """Grammar rule.

            expr_list: (expr_list COMMA)? expression
            """
            consume = None
            expr = None
            comma = None
            if isinstance(kid[0], ast.SubNodeList):
                consume = kid[0]
                comma = kid[1]
                expr = kid[2]
                new_kid = [*consume.kid, comma, expr]
            else:
                expr = kid[0]
                new_kid = [expr]
            valid_kid = [i for i in new_kid if isinstance(i, ast.Expr)]
            return self.nu(
                ast.SubNodeList[ast.Expr](
                    items=valid_kid,
                    kid=new_kid,
                )
            )

        def kw_expr_list(self, kid: list[ast.AstNode]) -> ast.SubNodeList[ast.KWPair]:
            """Grammar rule.

            kw_expr_list: (kw_expr_list COMMA)? kw_expr
            """
            consume = None
            expr = None
            comma = None
            if isinstance(kid[0], ast.SubNodeList):
                consume = kid[0]
                comma = kid[1]
                expr = kid[2]
                new_kid = [*consume.kid, comma, expr]
            else:
                expr = kid[0]
                new_kid = [expr]
            valid_kid = [i for i in new_kid if isinstance(i, ast.KWPair)]
            return self.nu(
                ast.SubNodeList[ast.KWPair](
                    items=valid_kid,
                    kid=new_kid,
                )
            )

        def kw_expr(self, kid: list[ast.AstNode]) -> ast.KWPair:
            """Grammar rule.

            kw_expr: any_ref EQ expression
            """
            if isinstance(kid[0], ast.NameSpec) and isinstance(kid[2], ast.Expr):
                return self.nu(
                    ast.KWPair(
                        key=kid[0],
                        value=kid[2],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def name_list(self, kid: list[ast.AstNode]) -> ast.SubNodeList[ast.Name]:
            """Grammar rule.

            name_list: (name_list COMMA)? NAME
            """
            consume = None
            name = None
            comma = None
            if isinstance(kid[0], ast.SubNodeList):
                consume = kid[0]
                comma = kid[1]
                name = kid[2]
                new_kid = [*consume.kid, comma, name]
            else:
                name = kid[0]
                new_kid = [name]
            valid_kid = [i for i in new_kid if isinstance(i, ast.Name)]
            return self.nu(
                ast.SubNodeList[ast.Name](
                    items=valid_kid,
                    kid=new_kid,
                )
            )

        def tuple_list(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.Expr | ast.KWPair]:
            """Grammar rule.

            tuple_list: expression COMMA expr_list COMMA kw_expr_list
                    | expression COMMA kw_expr_list
                    | expression COMMA expr_list
                    | expression COMMA
                    | kw_expr_list
            """
            chomp = [*kid]
            first_expr = None
            if isinstance(chomp[0], ast.SubNodeList):
                return self.nu(chomp[0])
            else:
                first_expr = chomp[0]
                chomp = chomp[2:]
            expr_list = []
            if len(chomp):
                expr_list = chomp[0].kid
                chomp = chomp[1:]
                if len(chomp):
                    chomp = chomp[1:]
                    expr_list = [*expr_list, *chomp[0].kid]
            expr_list = [first_expr, *expr_list]
            valid_kid = [i for i in expr_list if isinstance(i, (ast.Expr, ast.KWPair))]
            return self.nu(
                ast.SubNodeList[ast.Expr | ast.KWPair](
                    items=valid_kid,
                    kid=kid,
                )
            )

        def dict_val(self, kid: list[ast.AstNode]) -> ast.DictVal:
            """Grammar rule.

            dict_val: LBRACE ((kv_pair COMMA)* kv_pair)? RBRACE
            """
            ret = ast.DictVal(
                kv_pairs=[],
                kid=kid,
            )
            ret.kv_pairs = [i for i in kid if isinstance(i, ast.KVPair)]
            return self.nu(ret)

        def kv_pair(self, kid: list[ast.AstNode]) -> ast.KVPair:
            """Grammar rule.

            kv_pair: expression COLON expression
            """
            if isinstance(kid[0], ast.Expr) and isinstance(kid[2], ast.Expr):
                return self.nu(
                    ast.KVPair(
                        key=kid[0],
                        value=kid[2],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def list_compr(self, kid: list[ast.AstNode]) -> ast.ListCompr:
            """Grammar rule.

            list_compr: LSQUARE expression inner_compr+ RSQUARE
            """
            comprs = [i for i in kid if isinstance(i, ast.InnerCompr)]
            if isinstance(kid[1], ast.Expr):
                return self.nu(
                    ast.ListCompr(
                        out_expr=kid[1],
                        compr=comprs,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def gen_compr(self, kid: list[ast.AstNode]) -> ast.GenCompr:
            """Grammar rule.

            gen_compr: LSQUARE expression inner_compr+ RSQUARE
            """
            comprs = [i for i in kid if isinstance(i, ast.InnerCompr)]
            if isinstance(kid[1], ast.Expr):
                return self.nu(
                    ast.GenCompr(
                        out_expr=kid[1],
                        compr=comprs,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def set_compr(self, kid: list[ast.AstNode]) -> ast.SetCompr:
            """Grammar rule.

            set_compr: LSQUARE expression inner_compr+ RSQUARE
            """
            comprs = [i for i in kid if isinstance(i, ast.InnerCompr)]
            if isinstance(kid[1], ast.Expr) and isinstance(kid[2], ast.InnerCompr):
                return self.nu(
                    ast.SetCompr(
                        out_expr=kid[1],
                        compr=comprs,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def dict_compr(self, kid: list[ast.AstNode]) -> ast.DictCompr:
            """Grammar rule.

            dict_compr: LBRACE kv_pair inner_compr+ RBRACE
            """
            comprs = [i for i in kid if isinstance(i, ast.InnerCompr)]
            if isinstance(kid[1], ast.KVPair) and isinstance(kid[2], ast.InnerCompr):
                return self.nu(
                    ast.DictCompr(
                        kv_pair=kid[1],
                        compr=comprs,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def inner_compr(self, kid: list[ast.AstNode]) -> ast.InnerCompr:
            """Grammar rule.

            inner_compr: KW_ASYNC? KW_FOR atomic_chain KW_IN walrus_assign (KW_IF expression)?
            """
            chomp = [*kid]
            is_async = bool(
                isinstance(chomp[0], ast.Token) and chomp[0].name == Tok.KW_ASYNC
            )
            chomp = chomp[1:] if is_async else chomp
            chomp = chomp[1:]
            if isinstance(chomp[0], ast.Expr) and isinstance(chomp[2], ast.Expr):
                return self.nu(
                    ast.InnerCompr(
                        is_async=is_async,
                        target=chomp[0],
                        collection=chomp[2],
                        conditional=chomp[4]
                        if len(chomp) > 4 and isinstance(chomp[4], ast.Expr)
                        else None,
                        kid=chomp,
                    )
                )
            else:
                raise self.ice()

        def param_list(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.Expr | ast.KWPair]:
            """Grammar rule.

            param_list: expr_list COMMA kw_expr_list
                    | kw_expr_list
                    | expr_list
            """
            if len(kid) == 1:
                if isinstance(kid[0], ast.SubNodeList):
                    return self.nu(kid[0])
                else:
                    raise self.ice()
            elif isinstance(kid[0], ast.SubNodeList) and isinstance(
                kid[2], ast.SubNodeList
            ):
                valid_kid = [
                    i
                    for i in [*kid[0].items, *kid[2].items]
                    if isinstance(i, (ast.Expr, ast.KWPair))
                ]
                if len(valid_kid) == len(kid[0].items) + len(kid[2].items):
                    return self.nu(
                        ast.SubNodeList[ast.Expr | ast.KWPair](
                            items=valid_kid,
                            kid=kid,
                        )
                    )
                else:
                    raise self.ice()
            raise self.ice()

        def assignment_list(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.Assignment]:
            """Grammar rule.

            assignment_list: assignment_list COMMA assignment | assignment
            """
            consume = None
            assign = None
            comma = None
            if isinstance(kid[0], ast.SubNodeList):
                consume = kid[0]
                comma = kid[1]
                assign = kid[2]
                new_kid = [*consume.kid, comma, assign]
            else:
                assign = kid[0]
                new_kid = [assign]
            valid_kid = [i for i in new_kid if isinstance(i, ast.Assignment)]
            return self.nu(
                ast.SubNodeList[ast.Assignment](
                    items=valid_kid,
                    kid=new_kid,
                )
            )

        def arch_ref(self, kid: list[ast.AstNode]) -> ast.ArchRef:
            """Grammar rule.

            arch_ref: object_ref
                    | walker_ref
                    | edge_ref
                    | node_ref
                    | type_ref
            """
            if isinstance(kid[0], ast.ArchRef):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def node_ref(self, kid: list[ast.AstNode]) -> ast.ArchRef:
            """Grammar rule.

            node_ref: NODE_OP NAME
            """
            if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.NameSpec):
                return self.nu(
                    ast.ArchRef(
                        arch=kid[0],
                        name_ref=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def edge_ref(self, kid: list[ast.AstNode]) -> ast.ArchRef:
            """Grammar rule.

            edge_ref: EDGE_OP NAME
            """
            if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.NameSpec):
                return self.nu(
                    ast.ArchRef(
                        arch=kid[0],
                        name_ref=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def walker_ref(self, kid: list[ast.AstNode]) -> ast.ArchRef:
            """Grammar rule.

            walker_ref: WALKER_OP NAME
            """
            if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.NameSpec):
                return self.nu(
                    ast.ArchRef(
                        arch=kid[0],
                        name_ref=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def object_ref(self, kid: list[ast.AstNode]) -> ast.ArchRef:
            """Grammar rule.

            object_ref: OBJECT_OP name_ref
            """
            if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.NameSpec):
                return self.nu(
                    ast.ArchRef(
                        arch=kid[0],
                        name_ref=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def type_ref(self, kid: list[ast.AstNode]) -> ast.ArchRef:
            """Grammar rule.

            type_ref: TYPE_OP name_ref
            """
            if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.NameSpec):
                return self.nu(
                    ast.ArchRef(
                        arch=kid[0],
                        name_ref=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def enum_ref(self, kid: list[ast.AstNode]) -> ast.ArchRef:
            """Grammar rule.

            enum_ref: ENUM_OP NAME
            """
            if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.NameSpec):
                return self.nu(
                    ast.ArchRef(
                        arch=kid[0],
                        name_ref=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def ability_ref(self, kid: list[ast.AstNode]) -> ast.ArchRef:
            """Grammar rule.

            ability_ref: ABILITY_OP (special_ref | name_ref)
            """
            if isinstance(kid[0], ast.Token) and isinstance(kid[1], ast.NameSpec):
                return self.nu(
                    ast.ArchRef(
                        arch=kid[0],
                        name_ref=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def arch_or_ability_chain(self, kid: list[ast.AstNode]) -> ast.ArchRefChain:
            """Grammar rule.

            arch_or_ability_chain: arch_or_ability_chain? (ability_ref | arch_ref)
            """
            consume = None
            name = None
            if isinstance(kid[0], ast.SubNodeList):
                consume = kid[0]
                name = kid[1]
            else:
                name = kid[0]
            new_kid = [*consume.kid, name] if consume else [name]
            valid_kid = [i for i in new_kid if isinstance(i, ast.ArchRef)]
            if len(valid_kid) == len(new_kid):
                return self.nu(
                    ast.ArchRefChain(
                        archs=valid_kid,
                        kid=new_kid,
                    )
                )
            else:
                raise self.ice()

        def abil_to_arch_chain(self, kid: list[ast.AstNode]) -> ast.ArchRefChain:
            """Grammar rule.

            abil_to_arch_chain: arch_or_ability_chain? arch_ref
            """
            if len(kid) == 2:
                if isinstance(kid[1], ast.ArchRef) and isinstance(
                    kid[0], ast.ArchRefChain
                ):
                    return self.nu(
                        ast.ArchRefChain(
                            archs=[*(kid[0].archs), kid[1]],
                            kid=[*(kid[0].kid), kid[1]],
                        )
                    )
                else:
                    raise self.ice()
            elif isinstance(kid[0], ast.ArchRef):
                return self.nu(
                    ast.ArchRefChain(
                        archs=[kid[0]],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def arch_to_abil_chain(self, kid: list[ast.AstNode]) -> ast.ArchRefChain:
            """Grammar rule.

            arch_to_abil_chain: arch_or_ability_chain? ability_ref
            """
            if len(kid) == 2:
                if isinstance(kid[1], ast.ArchRef) and isinstance(
                    kid[0], ast.ArchRefChain
                ):
                    return self.nu(
                        ast.ArchRefChain(
                            archs=[*(kid[0].archs), kid[1]],
                            kid=[*(kid[0].kid), kid[1]],
                        )
                    )
                else:
                    raise self.ice()
            elif isinstance(kid[0], ast.ArchRef):
                return self.nu(
                    ast.ArchRefChain(
                        archs=[kid[0]],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def arch_to_enum_chain(self, kid: list[ast.AstNode]) -> ast.ArchRefChain:
            """Grammar rule.

            arch_to_enum_chain: arch_or_ability_chain? enum_ref
            """
            if len(kid) == 2:
                if isinstance(kid[1], ast.ArchRef) and isinstance(
                    kid[0], ast.ArchRefChain
                ):
                    return self.nu(
                        ast.ArchRefChain(
                            archs=[*(kid[0].archs), kid[1]],
                            kid=[*(kid[0].kid), kid[1]],
                        )
                    )
                else:
                    raise self.ice()
            elif isinstance(kid[0], ast.ArchRef):
                return self.nu(
                    ast.ArchRefChain(
                        archs=[kid[0]],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def edge_op_ref(self, kid: list[ast.AstNode]) -> ast.EdgeOpRef:
            """Grammar rule.

            edge_op_ref: edge_any
                       | edge_from
                       | edge_to
            """
            if isinstance(kid[0], ast.EdgeOpRef):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def edge_to(self, kid: list[ast.AstNode]) -> ast.EdgeOpRef:
            """Grammar rule.

            edge_to: ARROW_R_P1 expression (COLON filter_compare_list)? ARROW_R_P2
                   | ARROW_R
            """
            ftype = kid[1] if len(kid) >= 3 else None
            fcond = kid[3] if len(kid) >= 5 else None
            if (isinstance(ftype, ast.Expr) or ftype is None) and (
                isinstance(fcond, ast.SubNodeList) or fcond is None
            ):
                fcond = ast.FilterCompr(compares=fcond, kid=[fcond]) if fcond else None
                if fcond:
                    kid[3] = fcond
                return self.nu(
                    ast.EdgeOpRef(
                        filter_type=ftype,
                        filter_cond=fcond,
                        edge_dir=EdgeDir.OUT,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def edge_from(self, kid: list[ast.AstNode]) -> ast.EdgeOpRef:
            """Grammar rule.

            edge_from: ARROW_L_P1 expression (COLON filter_compare_list)? ARROW_L_P2
                     | ARROW_L
            """
            ftype = kid[1] if len(kid) >= 3 else None
            fcond = kid[3] if len(kid) >= 5 else None
            if (isinstance(ftype, ast.Expr) or ftype is None) and (
                isinstance(fcond, ast.SubNodeList) or fcond is None
            ):
                fcond = ast.FilterCompr(compares=fcond, kid=[fcond]) if fcond else None
                if fcond:
                    kid[3] = fcond
                return self.nu(
                    ast.EdgeOpRef(
                        filter_type=ftype,
                        filter_cond=fcond,
                        edge_dir=EdgeDir.IN,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def edge_any(self, kid: list[ast.AstNode]) -> ast.EdgeOpRef:
            """Grammar rule.

            edge_any: ARROW_L_P1 expression (COLON filter_compare_list)? ARROW_R_P2
                    | ARROW_BI
            """
            ftype = kid[1] if len(kid) >= 3 else None
            fcond = kid[3] if len(kid) >= 5 else None
            if (isinstance(ftype, ast.Expr) or ftype is None) and (
                isinstance(fcond, ast.SubNodeList) or fcond is None
            ):
                fcond = ast.FilterCompr(compares=fcond, kid=[fcond]) if fcond else None
                if fcond:
                    kid[3] = fcond
                return self.nu(
                    ast.EdgeOpRef(
                        filter_type=ftype,
                        filter_cond=fcond,
                        edge_dir=EdgeDir.ANY,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def connect_op(self, kid: list[ast.AstNode]) -> ast.ConnectOp:
            """Grammar rule.

            connect_op: connect_from
                      | connect_to
            """
            if isinstance(kid[0], ast.ConnectOp):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def disconnect_op(self, kid: list[ast.AstNode]) -> ast.DisconnectOp:
            """Grammar rule.

            disconnect_op: NOT edge_op_ref
            """
            if isinstance(kid[1], ast.EdgeOpRef):
                return self.nu(
                    ast.DisconnectOp(
                        edge_spec=kid[1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def connect_to(self, kid: list[ast.AstNode]) -> ast.ConnectOp:
            """Grammar rule.

            connect_to: CARROW_R_P1 expression (COLON kw_expr_list)? CARROW_R_P2
                      | CARROW_R
            """
            conn_type = kid[1] if len(kid) >= 3 else None
            conn_assign = kid[3] if len(kid) >= 5 else None
            if (isinstance(conn_type, ast.Expr) or conn_type is None) and (
                isinstance(conn_assign, ast.SubNodeList) or conn_assign is None
            ):
                conn_assign = (
                    ast.AssignCompr(assigns=conn_assign, kid=[conn_assign])
                    if conn_assign
                    else None
                )
                if conn_assign:
                    kid[3] = conn_assign
                return self.nu(
                    ast.ConnectOp(
                        conn_type=conn_type,
                        conn_assign=conn_assign,
                        edge_dir=EdgeDir.OUT,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def connect_from(self, kid: list[ast.AstNode]) -> ast.ConnectOp:
            """Grammar rule.

            connect_from: CARROW_L_P1 expression (COLON kw_expr_list)? CARROW_L_P2
                        | CARROW_L
            """
            conn_type = kid[1] if len(kid) >= 3 else None
            conn_assign = kid[3] if len(kid) >= 5 else None
            if (isinstance(conn_type, ast.Expr) or conn_type is None) and (
                isinstance(conn_assign, ast.SubNodeList) or conn_assign is None
            ):
                conn_assign = (
                    ast.AssignCompr(assigns=conn_assign, kid=[conn_assign])
                    if conn_assign
                    else None
                )
                if conn_assign:
                    kid[3] = conn_assign
                return self.nu(
                    ast.ConnectOp(
                        conn_type=conn_type,
                        conn_assign=conn_assign,
                        edge_dir=EdgeDir.IN,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def filter_compr(self, kid: list[ast.AstNode]) -> ast.FilterCompr:
            """Grammar rule.

            filter_compr: LPAREN EQ filter_compare_list RPAREN
            """
            if isinstance(kid[2], ast.SubNodeList):
                return self.nu(
                    ast.FilterCompr(
                        compares=kid[2],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def filter_compare_list(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.BinaryExpr]:
            """Grammar rule.

            filter_compare_list: (filter_compare_list COMMA)? filter_compare_item
            """
            consume = None
            expr = None
            comma = None
            if isinstance(kid[0], ast.SubNodeList):
                consume = kid[0]
                comma = kid[1]
                expr = kid[2]
                new_kid = [*consume.kid, comma, expr]
            else:
                expr = kid[0]
                new_kid = [expr]
            valid_kid = [i for i in new_kid if isinstance(i, ast.BinaryExpr)]
            return self.nu(
                ast.SubNodeList[ast.BinaryExpr](
                    items=valid_kid,
                    kid=new_kid,
                )
            )

        def filter_compare_item(self, kid: list[ast.AstNode]) -> ast.BinaryExpr:
            """Grammar rule.

            filter_compare_item: name_ref cmp_op expression
            """
            ret = self.binary_expr_unwind(kid)
            if isinstance(ret, ast.BinaryExpr):
                return self.nu(ret)
            else:
                raise self.ice()

        def assign_compr(self, kid: list[ast.AstNode]) -> ast.AssignCompr:
            """Grammar rule.

            filter_compr: LPAREN STAR_MUL kw_expr_list RPAREN
            """
            if isinstance(kid[2], ast.SubNodeList):
                return self.nu(
                    ast.AssignCompr(
                        assigns=kid[2],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def match_stmt(self, kid: list[ast.AstNode]) -> ast.MatchStmt:
            """Grammar rule.

            match_stmt: KW_MATCH expr_list LBRACE match_case_block+ RBRACE
            """
            cases = [i for i in kid if isinstance(i, ast.MatchCase)]
            if isinstance(kid[1], ast.Expr):
                return self.nu(
                    ast.MatchStmt(
                        target=kid[1],
                        cases=cases,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def match_case_block(self, kid: list[ast.AstNode]) -> ast.MatchCase:
            """Grammar rule.

            match_case_block: KW_CASE pattern_seq (KW_IF expression)? COLON statement_list
            """
            pattern = kid[1]
            guard = kid[3] if len(kid) > 4 else None
            stmts = kid[-1]
            if (
                isinstance(pattern, ast.MatchPattern)
                and isinstance(guard, (ast.Expr, type(None)))
                and isinstance(stmts, ast.SubNodeList)
            ):
                return self.nu(
                    ast.MatchCase(
                        pattern=pattern,
                        guard=guard,
                        body=stmts,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def pattern_seq(self, kid: list[ast.AstNode]) -> ast.MatchPattern:
            """Grammar rule.

            pattern_seq: (or_pattern | as_pattern)
            """
            if isinstance(kid[0], ast.MatchPattern):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def or_pattern(self, kid: list[ast.AstNode]) -> ast.MatchPattern:
            """Grammar rule.

            or_pattern: (pattern BW_OR)* pattern
            """
            if len(kid) == 1:
                if isinstance(kid[0], ast.MatchPattern):
                    return self.nu(kid[0])
                else:
                    raise self.ice()
            else:
                patterns = [i for i in kid if isinstance(i, ast.MatchPattern)]
                return self.nu(
                    ast.MatchOr(
                        patterns=patterns,
                        kid=kid,
                    )
                )

        def as_pattern(self, kid: list[ast.AstNode]) -> ast.MatchPattern:
            """Grammar rule.

            as_pattern: pattern KW_AS NAME
            """
            if isinstance(kid[0], ast.MatchPattern) and isinstance(
                kid[2], ast.NameSpec
            ):
                return self.nu(
                    ast.MatchAs(
                        pattern=kid[0],
                        name=kid[2],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def pattern(self, kid: list[ast.AstNode]) -> ast.MatchPattern:
            """Grammar rule.

            pattern: literal_pattern
                | capture_pattern
                | sequence_pattern
                | mapping_pattern
                | class_pattern
            """
            if isinstance(kid[0], ast.MatchPattern):
                return self.nu(kid[0])
            else:
                raise self.ice()

        def literal_pattern(self, kid: list[ast.AstNode]) -> ast.MatchPattern:
            """Grammar rule.

            literal_pattern: (INT | FLOAT | multistring)
            """
            if isinstance(kid[0], ast.Expr):
                return self.nu(
                    ast.MatchValue(
                        value=kid[0],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def singleton_pattern(self, kid: list[ast.AstNode]) -> ast.MatchPattern:
            """Grammar rule.

            singleton_pattern: (NULL | BOOL)
            """
            if isinstance(kid[0], (ast.Bool, ast.Null)):
                return self.nu(
                    ast.MatchSingleton(
                        value=kid[0],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def capture_pattern(self, kid: list[ast.AstNode]) -> ast.MatchPattern:
            """Grammar rule.

            capture_pattern: NAME
            """
            if (
                len(kid) == 1
                and isinstance(kid[0], ast.Name)
                and kid[0].sym_name == "_"
            ):
                return self.nu(
                    ast.MatchWild(
                        kid=kid,
                    )
                )
            if isinstance(kid[0], ast.NameSpec):
                return self.nu(
                    ast.MatchAs(
                        name=kid[0],
                        pattern=None,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def sequence_pattern(self, kid: list[ast.AstNode]) -> ast.MatchPattern:
            """Grammar rule.

            sequence_pattern: LSQUARE list_inner_pattern (COMMA list_inner_pattern)* RSQUARE
                            | LPAREN list_inner_pattern (COMMA list_inner_pattern)* RPAREN
            """
            patterns = [i for i in kid if isinstance(i, ast.MatchPattern)]
            return self.nu(
                ast.MatchSequence(
                    values=patterns,
                    kid=kid,
                )
            )

        def mapping_pattern(self, kid: list[ast.AstNode]) -> ast.MatchMapping:
            """Grammar rule.

            mapping_pattern: LBRACE (dict_inner_pattern (COMMA dict_inner_pattern)*)? RBRACE
            """
            patterns = [
                i for i in kid if isinstance(i, (ast.MatchKVPair, ast.MatchStar))
            ]
            return self.nu(
                ast.MatchMapping(
                    values=patterns,
                    kid=kid,
                )
            )

        def list_inner_pattern(self, kid: list[ast.AstNode]) -> ast.MatchPattern:
            """Grammar rule.

            list_inner_pattern: (pattern_seq | STAR_MUL NAME)
            """
            if isinstance(kid[0], ast.MatchPattern):
                return self.nu(kid[0])
            elif isinstance(kid[-1], ast.Name):
                return self.nu(
                    ast.MatchStar(
                        is_list=True,
                        name=kid[-1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def dict_inner_pattern(
            self, kid: list[ast.AstNode]
        ) -> ast.MatchKVPair | ast.MatchStar:
            """Grammar rule.

            dict_inner_pattern: (pattern_seq COLON pattern_seq | STAR_POW NAME)
            """
            if isinstance(kid[0], ast.MatchPattern) and isinstance(
                kid[2], ast.MatchPattern
            ):
                return self.nu(
                    ast.MatchKVPair(
                        key=kid[0],
                        value=kid[2],
                        kid=kid,
                    )
                )
            elif isinstance(kid[-1], ast.Name):
                return self.nu(
                    ast.MatchStar(
                        is_list=False,
                        name=kid[-1],
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def class_pattern(self, kid: list[ast.AstNode]) -> ast.MatchArch:
            """Grammar rule.

            class_pattern: NAME LPAREN kw_pattern_list? RPAREN
                        | NAME LPAREN pattern_list (COMMA kw_pattern_list)? RPAREN
            """
            name = kid[0]
            first = kid[2]
            second = kid[4] if len(kid) > 4 else None
            arg = (
                first
                if isinstance(first, ast.SubNodeList)
                and isinstance(first.items[0], ast.MatchPattern)
                else None
            )
            kw = (
                second
                if isinstance(second, ast.SubNodeList)
                and isinstance(second.items[0], ast.MatchKVPair)
                else first
                if isinstance(first, ast.SubNodeList)
                and isinstance(first.items[0], ast.MatchKVPair)
                else None
            )
            if isinstance(name, ast.NameSpec):
                return self.nu(
                    ast.MatchArch(
                        name=name,
                        arg_patterns=arg,
                        kw_patterns=kw,
                        kid=kid,
                    )
                )
            else:
                raise self.ice()

        def pattern_list(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.MatchPattern]:
            """Grammar rule.

            pattern_list: (pattern_list COMMA)? pattern_seq
            """
            consume = None
            pattern = None
            comma = None
            if isinstance(kid[0], ast.SubNodeList):
                consume = kid[0]
                comma = kid[1]
                pattern = kid[2]
            else:
                pattern = kid[0]
            new_kid = [*consume.kid, comma, pattern] if consume else [pattern]
            valid_kid = [i for i in new_kid if isinstance(i, ast.MatchPattern)]
            return ast.SubNodeList[ast.MatchPattern](
                items=valid_kid,
                kid=kid,
            )

        def kw_pattern_list(
            self, kid: list[ast.AstNode]
        ) -> ast.SubNodeList[ast.MatchKVPair]:
            """Grammar rule.

            kw_pattern_list: (kw_pattern_list COMMA)? named_ref EQ pattern_seq
            """
            consume = None
            name = None
            eq = None
            value = None
            comma = None
            if isinstance(kid[0], ast.SubNodeList):
                consume = kid[0]
                comma = kid[1]
                name = kid[2]
                eq = kid[3]
                value = kid[4]
                if not isinstance(name, ast.NameSpec) or not isinstance(
                    value, ast.MatchPattern
                ):
                    raise self.ice()
                new_kid = [
                    *consume.kid,
                    comma,
                    ast.MatchKVPair(key=name, value=value, kid=[name, eq, value]),
                ]
            else:
                name = kid[0]
                eq = kid[1]
                value = kid[2]
                if not isinstance(name, ast.NameSpec) or not isinstance(
                    value, ast.MatchPattern
                ):
                    raise self.ice()
                new_kid = [
                    ast.MatchKVPair(key=name, value=value, kid=[name, eq, value])
                ]
            if isinstance(name, ast.NameSpec) and isinstance(value, ast.MatchPattern):
                valid_kid = [i for i in new_kid if isinstance(i, ast.MatchKVPair)]
                return ast.SubNodeList[ast.MatchKVPair](
                    items=valid_kid,
                    kid=new_kid,
                )
            else:
                raise self.ice()

        def __default_token__(self, token: jl.Token) -> ast.Token:
            """Token handler."""
            ret_type = ast.Token
            if token.type == Tok.KWESC_NAME:
                return self.nu(
                    ast.Name(
                        file_path=self.parse_ref.mod_path,
                        name=token.type,
                        value=token.value[2:],
                        line=token.line if token.line is not None else 0,
                        col_start=token.column if token.column is not None else 0,
                        col_end=token.end_column if token.end_column is not None else 0,
                        pos_start=token.start_pos if token.start_pos is not None else 0,
                        pos_end=token.end_pos if token.end_pos is not None else 0,
                        is_kwesc=True,
                        kid=[],
                    )
                )
            elif token.type == Tok.NAME:
                ret_type = ast.Name
            elif token.type == Tok.SEMI:
                ret_type = ast.Semi
            elif token.type == Tok.NULL:
                ret_type = ast.Null
            elif token.type == Tok.FLOAT:
                ret_type = ast.Float
            elif token.type in [Tok.INT, Tok.INT, Tok.HEX, Tok.BIN, Tok.OCT]:
                ret_type = ast.Int
            elif token.type in [
                Tok.STRING,
                Tok.FSTR_BESC,
                Tok.FSTR_PIECE,
                Tok.DOC_STRING,
            ]:
                ret_type = ast.String
            elif token.type == Tok.BOOL:
                ret_type = ast.Bool
            elif token.type == Tok.PYNLINE and isinstance(token.value, str):
                token.value = token.value.replace("::py::", "")
            return self.nu(
                ret_type(
                    file_path=self.parse_ref.mod_path,
                    name=token.type,
                    value=token.value,
                    line=token.line if token.line is not None else 0,
                    col_start=token.column if token.column is not None else 0,
                    col_end=token.end_column if token.end_column is not None else 0,
                    pos_start=token.start_pos if token.start_pos is not None else 0,
                    pos_end=token.end_pos if token.end_pos is not None else 0,
                    kid=[],
                )
            )
