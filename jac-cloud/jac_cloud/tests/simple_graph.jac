"""Example of simple walker walking nodes."""
import:py from jac_cloud.core.context {JaseciContext}
import:py from jac_cloud.jaseci.models {User as BaseUser, NO_PASSWORD}
import:py from jac_cloud.jaseci.utils {SendGridEmailer}
import:py from dataclasses {dataclass}
import:py from fastapi {UploadFile}
import:py from typing {Any}

class User:BaseUser: {
    has name: str;

    class Collection:BaseUser.Collection: {
        ::py::
        @classmethod
        def __document__(cls, doc) -> "User":
            return User(
                id=doc.pop("_id"),
                email=doc.pop("email"),
                password=doc.pop("password", None) or NO_PASSWORD,
                root_id=doc.pop("root_id"),
                **doc,
            )
        ::py::
    }

    static can send_verification_code(code: str, email: str) -> None {
        SendGridEmailer.send_verification_code(code, email);
    }

    static can send_reset_code(code: str, email: str) -> None {
        SendGridEmailer.send_reset_code(code, email);
    }

    static can sso_mapper(open_id: object) -> dict[str, object] {
        return {
            "name": f"{open_id.first_name} {open_id.last_name}"
        };
    }

    static can system_admin_default() -> dict[str, object] {
        return {
            "name": f"System Administrator"
        };
    }
}

enum Enum {
    A = "a",
    B = "b",
    C = "c"
}

node A {
    has val: int;
}

node B {
    has val: str | int;
}

node C {
    has val: Any;
}

obj Child {
    has val: int, arr: list[int], json: dict[str, int], enum_field: Enum;
}

obj Parent:Child: {
    has child: Child;
}

node Nested {
    has val: int, arr: list[int], json: dict[str, int], parent: Parent, enum_field: Enum;
}

walker create_graph {
    can enter_root with `root entry {
        a = A(val=0);
        b = B(val=1);
        c = C(val=2);
        here ++> a;
        a ++> b;
        b ++> c;

        report here;
        report a;
        report b;
        report c;
    }
}

walker traverse_graph {
    can enter with `root entry {
        report here;
        visit [-->];
    }

    can enter_A with A entry {
        report here;
        visit [-->];
    }

    can enter_B with B entry {
        report here;
        visit [-->];
    }

    can enter_C with C entry {
        report here;
    }
}

walker detach_node {
    can enter with `root entry {
        visit [-->];
    }

    can enter_A with A entry {
        visit [-->];
    }

    can enter_B with B entry {
        return here del --> [-->];
    }
}

walker update_graph {
    can enter with `root entry {
        report here;
        visit [-->];
    }

    can enter_A with A entry {
        here.val = 1;
        report here;
        visit [-->];
    }

    can enter_B with B entry {
        here.val = 2;
        report here;
        visit [-->];
    }
}

walker create_nested_node {
    can enter_root with `root entry {
        n = Nested(
            val=0,
            arr=[],
            json={},
            parent=Parent(
                val=1,
                arr=[1],
                json={"a": 1},
                child=Child(
                    val=2,
                    arr=[1, 2],
                    json={"a": 1, "b": 2},
                    enum_field = Enum.C
                ),
                enum_field = Enum.B
            ),
            enum_field = Enum.A
        );
        here ++> n;
        return n;
    }
}

walker update_nested_node {
    can enter_root with `root entry {
        nested = [-->(`?Nested)][0];
        nested.parent.child.json["c"] = 3;
        nested.parent.child.arr.append(3);
        nested.parent.child.val = 3;
        nested.parent.child.enum_field = Enum.A;
        nested.parent.json["b"] = 2;
        nested.parent.arr.append(2);
        nested.parent.val = 2;
        nested.parent.enum_field = Enum.C;
        nested.json["a"] = 1;
        nested.arr.append(1);
        nested.val = 1;
        nested.enum_field = Enum.B;
        return nested;
    }

    can enter_nested with Nested entry {
        here.parent.child.json["c"] = 3;
        here.parent.child.arr.append(3);
        here.parent.child.val = 3;
        here.parent.child.enum_field = Enum.A;
        here.parent.json["b"] = 2;
        here.parent.arr.append(2);
        here.parent.val = 2;
        here.parent.enum_field = Enum.C;
        here.json["a"] = 1;
        here.arr.append(1);
        here.val = 1;
        here.enum_field = Enum.B;
        return here;
    }
}

walker detach_nested_node {
    can enter_root with `root entry {
        report here del--> [-->(`?Nested)];
    }
}

walker visit_nested_node {
    can enter_root with `root entry {
        nesteds = [-->(`?Nested)];
        if nesteds {
            report [-->(`?Nested)][0];
        } else {
            report nesteds;
        }
    }

    can enter_nested with Nested entry {
        report here;
    }
}

walker delete_nested_node {
    can enter_root with `root entry {
        nested = [-->(`?Nested)][0];
        Jac.destroy(nested);
        # nested.__jac__.apply();

        report [-->(`?Nested)];
    }
}

walker delete_nested_edge {
    can enter_root with `root entry {
        nested_edge = :e:[-->][0];
        Jac.destroy(nested_edge);

        report [-->(`?Nested)];
    }
}

walker allow_other_root_access {
    has root_id: str, level: int | str = 0, via_all: bool = False;

    can enter_root with `root entry {
        if self.via_all {
            Jac.unrestrict(here, self.level);
        } else {
            import:py from jac_cloud.core.architype {BaseAnchor}
            Jac.allow_root(here, BaseAnchor.ref(self.root_id), self.level);
        }
    }

    can enter_nested with Nested entry {
        if self.via_all {
            Jac.unrestrict(here, self.level);
        } else {
            import:py from jac_cloud.core.architype {BaseAnchor}
            Jac.allow_root(here, BaseAnchor.ref(self.root_id), self.level);
        }
    }
}

walker disallow_other_root_access {
    has root_id: str, via_all: bool = False;

    can enter_root with `root entry {
        if self.via_all {
            Jac.restrict(here);
        } else {
            import:py from jac_cloud.core.architype {BaseAnchor}
            Jac.disallow_root(here, BaseAnchor.ref(self.root_id));
        }
    }

    can enter_nested with Nested entry {
        if self.via_all {
            Jac.restrict(here);
        } else {
            import:py from jac_cloud.core.architype {BaseAnchor}
            Jac.disallow_root(here, BaseAnchor.ref(self.root_id));
        }
    }
}

#################################################################
#                    ENDPOINT CUSTOMIZATIONS                    #
#################################################################

walker post_no_body {}

walker post_with_body {
    has a: str;
}

walker get_no_body {
    class __specs__ {
        has methods: list = ["get"];
    }
}

walker get_with_query {
    has a: str;

    class __specs__ {
        has methods: list = ["get"], as_query: list = ["a"];
    }
}

walker get_all_query {
    has a: str;
    has b: str;

    class __specs__ {
        has methods: list = ["get"], as_query: list = "*", auth: bool = False;
    }
}

walker post_path_var {
    has a: str;

    class __specs__ {
        has path: str = "/{a}", methods: list = ["post", "get"];
    }
}

walker post_all_excluded {
    has a: str = "";
    has b: str = "";

    class __specs__ {
        has excluded: str = "*", auth: bool = False;
    }
}

walker post_partial_excluded {
    has a: str;
    has b: str = "";

    class __specs__ {
        has excluded: str = "b";
    }
}

walker combination1 {
    has a: str;
    has b: str;
    has c: str;
    has d: str = "";

    class __specs__ {
        has methods: list = ["post", "get"], as_query: list = ["a", "b"], excluded: str = "d";
    }
}


walker combination2 {
    has a: str;
    has b: str;
    has c: str;
    has d: str = "";
    has e: str = "";

    class __specs__ {
        has path: str = "/{a}",
        methods: list = ["post", "get", "put", "patch", "delete", "head", "trace", "options"],
        as_query: list = ["b"],
        excluded: list = ["d", "e"];
    }
}

walker post_with_file {
    has single: UploadFile;
    has multiple: list[UploadFile];
    has singleOptional: UploadFile | None = None;


    can enter with `root entry {
        print(self.single);
        print(self.multiple);
        print(self.singleOptional);

        multiple = [];
        for file in self.multiple {
            multiple.append({
                "name": file.filename,
                "content_type": file.content_type,
                "size": file.size,
            });
        }

        single = {
            "single": {
                "name": self.single.filename,
                "content_type": self.single.content_type,
                "size": self.single.size,
            }
        };

        report {
            "single": single,
            "multiple": multiple,
            "singleOptional": self.singleOptional
        };
    }

    class __specs__ {}
}

walker post_with_body_and_file {
    has val: int;
    has single: UploadFile;
    has multiple: list[UploadFile];
    has optional_val: int = 0;

    can enter with `root entry {
        print(self.val);
        print(self.optional_val);
        print(self.single);
        print(self.multiple);
    }

    class __specs__ {
        has auth: bool = False;
    }
}

walker custom_status_code {
    has status: int;

    can enter with `root entry {
        Jac.get_context().status = self.status;
    }
}

walker different_return {

    can enter1 with `root entry {
    }

    can enter2 with A entry -> int {
    }

    can enter3 with B entry -> str {
    }

    can enter4 with C entry -> bool {
    }

    can enter5 with Nested entry -> Parent {
    }

    can enter6 with `root entry -> list | dict {
    }

    class __specs__ {
        has auth: bool = False;
    }
}

:walker:detach_nested_node:can:enter_root {
    return here del--> [-->(`?Nested)];
}

:walker:visit_nested_node:can:enter_root {
    nesteds = [-->(`?Nested)];
    if nesteds {
        return [-->(`?Nested)][0];
    } else {
        return nesteds;
    }
}

:walker:visit_nested_node:can:enter_nested {
    return here;
}

walker manual_create_nested_node {
    can enter_root with `root entry {
        n = Nested(
            val=0,
            arr=[],
            json={},
            parent=Parent(
                val=1,
                arr=[1],
                json={"a": 1},
                child=Child(
                    val=2,
                    arr=[1, 2],
                    json={"a": 1, "b": 2},
                    enum_field = Enum.C
                ),
                enum_field = Enum.B
            ),
            enum_field = Enum.A
        );
        here ++> n;
        here.__jac__.apply();

        # simulate no auto save
        jsrc = Jac.get_context().mem;
        jsrc.__mem__.clear();
        jsrc.__gc__.clear();

        return n;
    }
}

walker manual_update_nested_node {
    can enter_root with `root entry {
        nested = [-->(`?Nested)][0];
        nested.parent.child.json["c"] = 3;
        nested.parent.child.arr.append(3);
        nested.parent.child.val = 3;
        nested.parent.child.enum_field = Enum.A;
        nested.parent.json["b"] = 2;
        nested.parent.arr.append(2);
        nested.parent.val = 2;
        nested.parent.enum_field = Enum.C;
        nested.json["a"] = 1;
        nested.arr.append(1);
        nested.val = 1;
        nested.enum_field = Enum.B;
        nested.__jac__.apply();

        # simulate no auto save
        jsrc = Jac.get_context().mem;
        jsrc.__mem__.clear();
        jsrc.__gc__.clear();

        return nested;
    }
}

walker manual_detach_nested_node {
    can enter_root with `root entry {
        nested = [-->(`?Nested)][0];
        detached = here del--> [-->(`?Nested)];
        Jac.destroy(nested);
        nested.__jac__.apply();

        # simulate no auto save
        jsrc = Jac.get_context().mem;
        jsrc.__mem__.clear();
        jsrc.__gc__.clear();

        return detached;
    }
}

walker manual_delete_nested_node {
    can enter_root with `root entry {
        nested = [-->(`?Nested)][0];
        Jac.destroy(nested);
        nested.__jac__.apply();

        # simulate no auto save
        jsrc = Jac.get_context().mem;
        jsrc.__mem__.clear();
        jsrc.__gc__.clear();

        report [-->(`?Nested)];
    }
}

walker manual_delete_nested_edge {
    can enter_root with `root entry {
        nested_edge = :e:[-->][0];
        Jac.destroy(nested_edge);
        nested_edge.__jac__.apply();

        # simulate no auto save
        jsrc = Jac.get_context().mem;
        jsrc.__mem__.clear();
        jsrc.__gc__.clear();

        report [-->(`?Nested)];
    }
}

walker custom_report {
    can enter1 with `root entry {
        report 1;
        report 2;
        report 3;

        Jac.report({
            "testing": 1
        }, True);
    }

    class __specs__ {
        has auth: bool = False;
    }
}

##################################################################
#                           FOR PURGER                           #
##################################################################

node D {
    has id: int;
}

node E {
    has id: int;
}

node F {
    has id: int;
}

node G {
    has id: int;
}

node H {
    has id: int;
}


walker populate_graph {
    can setup1 with `root entry {
        for i in range(2) {
            here ++> D(id=i);
        }
        visit [-->];
    }

    can setup2 with D entry {
        for i in range(2) {
            here ++> E(id=i);
        }
        visit [-->];
    }

    can setup3 with E entry {
        for i in range(2) {
            here ++> F(id=i);
        }
        visit [-->];
    }

    can setup4 with F entry {
        for i in range(2) {
            here ++> G(id=i);
        }
        visit [-->];
    }

    can setup5 with G entry {
        for i in range(2) {
            here ++> H(id=i);
        }
        visit [-->];
    }
}

walker traverse_populated_graph {
    can enter1 with `root entry {
        report here;
        visit [-->];
    }

    can enter2 with D entry {
        report here;
        visit [-->];
    }

    can enter3 with E entry {
        report here;
        visit [-->];
    }

    can enter4 with F entry {
        report here;
        visit [-->];
    }

    can enter5 with G entry {
        report here;
        visit [-->];
    }

    can enter6 with H entry {
        report here;
        visit [-->];
    }
}

walker purge_populated_graph {
    can purge with `root entry {
        report Jac.reset_graph();
    }
}

walker check_populated_graph {
    can enter with `root entry {
        import:py from jac_cloud.core.architype {NodeAnchor, EdgeAnchor, WalkerAnchor}

        id = here.__jac__.id;
        count = NodeAnchor.Collection.count({"$or": [{"_id": id}, {"root": id}]});
        count += EdgeAnchor.Collection.count({"root": id});
        count += WalkerAnchor.Collection.count({"root": id});

        report count;
    }
}

walker check_memory_sync {
    has other_node_id: str;
    can enter with `root entry {
        import:py from jac_cloud.core.architype {NodeAnchor}
        n = &(self.other_node_id);
        n.val = 4;
        report n;
    }
}

#################################################################
#                        SAVABLE OBJECT                         #
#################################################################

obj SavableObject {
    has val: int, arr: list[int], json: dict[str, int], parent: Parent, enum_field: Enum;
}

walker create_custom_object {
    can enter1 with `root entry {
        o = SavableObject(
            val=0,
            arr=[],
            json={},
            parent=Parent(
                val=1,
                arr=[1],
                json={"a": 1},
                child=Child(
                    val=2,
                    arr=[1, 2],
                    json={"a": 1, "b": 2},
                    enum_field = Enum.C
                ),
                enum_field = Enum.B
            ),
            enum_field = Enum.A
        );
        Jac.save(o);
        report o;
    }
}

walker get_custom_object {
    has object_id: str;

    can enter1 with `root entry {
        try {
            report &(self.object_id);
        } except Exception as e {
            report None;
        }
    }
}

walker update_custom_object {
    has object_id: str;

    can enter1 with `root entry {
        savable_object = &(self.object_id);
        savable_object.parent.child.json["c"] = 3;
        savable_object.parent.child.arr.append(3);
        savable_object.parent.child.val = 3;
        savable_object.parent.child.enum_field = Enum.A;
        savable_object.parent.json["b"] = 2;
        savable_object.parent.arr.append(2);
        savable_object.parent.val = 2;
        savable_object.parent.enum_field = Enum.C;
        savable_object.json["a"] = 1;
        savable_object.arr.append(1);
        savable_object.val = 1;
        savable_object.enum_field = Enum.B;
        report savable_object;
    }
}

walker delete_custom_object {
    has object_id: str;

    can enter1 with `root entry {
        import:py from jac_cloud.core.architype {BaseAnchor}
        Jac.destroy(&(self.object_id));

        # This is similar to
        #
        # Jac.destroy(BaseAnchor.ref(self.object_id));
        #
        # The only difference is BaseAnchor.ref doesn't
        # load the actual object and just use it as reference
    }
}