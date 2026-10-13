Databases written by a build from before a column type change rewrote its
layers, read by tests/retype_fixture_test.rs. The current build cannot
write them: its type change casts every layer and waits for a commit's
marker.

Generator: commit 719a18a8 (the parent of #313), program generator.rs.txt
placed at examples/retype_fixtures.rs and run as

    cargo run --example retype_fixtures --features test-failpoints -- <out>

The program runs its steps in this order; db.lock files were left out. The
first four fixtures come from a run of the program before step 5 was added;
hot_colliding comes from a second run of the program as it stands, whose
other outputs were not kept.

1. old_typed_volumes: k INTEGER UNIQUE, rows (1, 7) and (2, 8) sealed,
   MODIFY COLUMN k TEXT, closed. The volume holds INTEGER under a TEXT schema.
2. colliding_volumes: k TEXT UNIQUE, '7' sealed, '07' sealed in a second
   volume, MODIFY COLUMN k INTEGER, closed. Both cast to 7.
3. row_alter_marker: row (1, 7), then row (2, 8) whose commit runs MODIFY
   COLUMN k TEXT before its marker is written; the files copied as a crash
   would leave them. The WAL holds row 2, the change, row 2's marker.
4. text_integer_text: k TEXT, row (1, '007'), MODIFY COLUMN k INTEGER,
   MODIFY COLUMN k TEXT, files copied as a crash would leave them.
5. hot_colliding: k TEXT UNIQUE, rows (1, '7') and (2, '07') never sealed,
   MODIFY COLUMN k INTEGER, files copied as a crash would leave them. Replay
   meets two rows whose keys the change makes equal.

unversioned_renames comes from commit 66b31323 (the parent of #177, before
renames carried a version), program rename_generator.rs.txt placed at
examples/rename_fixtures.rs and run as

    cargo run --example rename_fixtures -- <out>

Their renames carry no version, so which volume column a name stands for
is not known in general. The current build opens them, refuses a change
of a column's type with every file kept, and refuses to open the one that
needs its volumes rewritten.

6. unversioned_renames: a = 5, b = 6, k = 7 sealed, then RENAME b TO c,
   a TO b, b TO d, checkpointed and closed.
7. sealed_between_renames: a = 5, k = 7, RENAME a TO b, sealed, then
   RENAME b TO c, checkpointed and closed.
8. dropped_before_seal: RENAME a TO c, DROP COLUMN c, RENAME b TO a,
   row (1, 6, 7) sealed, then ADD COLUMN c INTEGER DEFAULT 9,
   checkpointed and closed.
9. passed_name_added_again: k = 7, a = 5 sealed, then RENAME a TO b,
   b TO c, c TO d, DROP COLUMN d, ADD COLUMN c INTEGER DEFAULT 9,
   checkpointed and closed.
10. dropped_after_reopen: RENAME b TO c, a = 5, c = 6 sealed, closed and
    reopened (which restarts the schema epoch), then DROP COLUMN a,
    RENAME c TO a, ADD COLUMN c INTEGER DEFAULT 9, checkpointed and
    closed. The drop records a version below the seal's.
11. unversioned_old_typed: a = 5, k = 7, RENAME a TO b, sealed, then
    MODIFY COLUMN k TEXT, checkpointed and closed. The volume holds k as
    INTEGER under a TEXT schema.
12. name_dropped_after_reopen: x = 5, k = 7, y = 8 sealed at version 2,
    closed and reopened (which restarts the schema epoch), then DROP
    COLUMN x at version 1 and ADD COLUMN x INTEGER DEFAULT 9, checkpointed
    and closed. It has no rename; the drop's version says it came before
    the seal, so the volume's x is read for the added one. The current
    build refuses a change of type that would rewrite that volume.

unversioned_renames comes from a run of the program before step 7 was
added, sealed_between_renames from a run before step 8 was added,
dropped_before_seal from a run before step 9 was added,
passed_name_added_again from a run before step 10 was added,
dropped_after_reopen from a run before step 11 was added,
unversioned_old_typed from a run before step 12 was added, and
name_dropped_after_reopen from a seventh run; the other outputs of the
later runs were not kept.

SHA-256 of every file:

42a9d1b7dd9c2de29ca18704a2776fe0b563d00693dd6cc31663e91ff1a46938  colliding_volumes/volumes/t/manifest.bin
563328c57c4b9dfe18f1178b9ab4a464cf5602048a87a50f111e0f932ea2b1cd  colliding_volumes/volumes/t/vol_00065daee83883d8.vol
d4a6250ed933c4bba21f3367319888cd2cc512b7481471780636e1ee5ca72741  colliding_volumes/volumes/t/vol_00065daee838d6a4.vol
7b62df827b8d7243f84030e27d75593ad355db59dc0231763f9615b3ffcd58d2  colliding_volumes/wal/checkpoint.meta
37d3d5bfc390c879d51010d086dfe459a7dcfda7169d32155ab383f552be0eb5  colliding_volumes/wal/wal_00000002-20261013-014102-lsn-16.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  colliding_volumes/wal/wal_00000003-20261013-014102-lsn-22.log
9bde7c2f7b4915b50ee08e302e219fb6af6cbd6f42915c2a230d895072870c53  old_typed_volumes/volumes/t/manifest.bin
8ff61a393ff5235f486692b9a18d17aeeb7e7d8cf5615e5863d6ac5e45624056  old_typed_volumes/volumes/t/vol_00065daee836f7ee.sidx
4020e8012637b23b20f1e2ae5e74ffc36610874bbcfb41e6a685673920784505  old_typed_volumes/volumes/t/vol_00065daee836f7ee.vol
c0693a08388e48e8c67a896001b9d4fb396a17097d57afff6f1f25565c91a53c  old_typed_volumes/wal/checkpoint.meta
1da779bb268f8950fdaf6c910dc9a1560b57c0e8be6b80490336b2a854d1dd63  old_typed_volumes/wal/wal_00000001-20261013-014101-lsn-11.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  old_typed_volumes/wal/wal_00000002-20261013-014101-lsn-17.log
561e98c283925f74fe3d21263673f5d67b5b49f1608040e0e0a7bb728187ac84  hot_colliding/wal/wal-20261013-022925-lsn-0.log
b4a9070ce26dd4f7daa817281d2f211b2529b79680b0dfd5d93b70d24b113872  row_alter_marker/wal/wal-20261013-014102-lsn-0.log
fcf62734af863be65c4a38fe2f811ee20e71915b126c4f56fb26a9109fef2324  text_integer_text/wal/wal-20261013-014102-lsn-0.log
0154ff9ee823c9a9a13e0c198bcfbef63a08e5400e1fdeb6fd00fedf26dd4379  unversioned_renames/volumes/t/manifest.bin
a1a5ab25e8b28d46452b3e81b093315d27e6f3632b503ce1a5c664f82af4bd0c  unversioned_renames/volumes/t/vol_00065db6c18a238c.vol
e4f06aa1d2bc558742ad31f7ba21a23276cbd35b84ea092a03b3bf8b04a7fdcc  unversioned_renames/wal/checkpoint.meta
7c10ba40e1526bac19cac601cd401f809858e80d21c3afca4d6b229196fe4fd7  unversioned_renames/wal/wal-20261013-110252-lsn-14.log
3725b5a75a0514394f8cb9e92444fcca97765fe13b9f4f5c14c03ad2dc58048c  sealed_between_renames/volumes/t/manifest.bin
28af276e92d1be1c2797edd53c5136137819542738c613085359237d59ca289a  sealed_between_renames/volumes/t/vol_00065db7769ea11f.vol
5ee548699385226391a2ed17496e32e0dfa8ada9a40c304f3e3abafd80822e5f  sealed_between_renames/wal/checkpoint.meta
0aa2449f14eef7b8042652c204ed6daf141d8597a672291c46bb57792f0fd4ab  sealed_between_renames/wal/wal-20261013-115330-lsn-12.log
8d40f9f688b67e966284786b8f78940a2a7557d63754fa6be21a5a6a21ee4ed0  dropped_before_seal/volumes/t/manifest.bin
d7bd3e5d7e86c6f4942e4bbc1143ff13c048a2a82059a130094a702450251a93  dropped_before_seal/volumes/t/vol_00065db832aab5e0.vol
49d3e3b07ba530ca9e02675275aa9064e22bece448b3782c9c49e11c96b78ffd  dropped_before_seal/wal/checkpoint.meta
00bbfe77b100543fb6e0869df8078ce06c2b6675441bb7945082828b73a10f34  dropped_before_seal/wal/wal-20261013-124605-lsn-16.log
e3f5285881aa05969eb21b9920537b24705a15a2a256b44dbcf6c196221b7055  passed_name_added_again/volumes/t/manifest.bin
d2370625b2925c2449301b9b92746fd39cb471c522973021ce19a29b02fa39c7  passed_name_added_again/volumes/t/vol_00065db8b8fa7790.vol
3d5e74ae58ebc06566bad517ac07766dd17953a0c8f28424856cb107baba791d  passed_name_added_again/wal/checkpoint.meta
3593e7249ab33cbe5bf480f6096952cdf43058443835695b5e9841ae574a38ee  passed_name_added_again/wal/wal-20261013-132339-lsn-18.log
0d3d93a2812e64bd271f00443b54fb91dc44c52fd84c91eb03da8ac42d111cfb  dropped_after_reopen/volumes/t/manifest.bin
b6467a91bac3cfce50676bb0c81c8caebf41349484a4e96957fc3050e7605a56  dropped_after_reopen/volumes/t/vol_00065db9fda524bb.vol
4cca28c87af83d0dd806a3738f4811a4dc73fac1164621548fe4f733e1c0b55b  dropped_after_reopen/wal/checkpoint.meta
0c46941facc61d9f5179a36455c28a01f8371db50bfebf192b20f45ffb7c5d45  dropped_after_reopen/wal/wal-20261013-145426-lsn-18.log
ed4761c8833ba76af137d7f18231e7937ad9c723c4eef64df276669a0f152304  unversioned_old_typed/volumes/t/manifest.bin
28af276e92d1be1c2797edd53c5136137819542738c613085359237d59ca289a  unversioned_old_typed/volumes/t/vol_00065dba57d16c9f.vol
e9a9e8d2b4a2ad5cb14e5555ad08c9df00feac3ce4bf4de299e62d639241a8ae  unversioned_old_typed/wal/checkpoint.meta
a80a420a65d6b0cbdd629e8227327ea2c91168663c12e043e2fcb8c9a6162b46  unversioned_old_typed/wal/wal-20261013-151939-lsn-12.log
6792b6af1feb91c251f0c1b9905defb6575208260cbaca82161bd95d1ffe5c5a  name_dropped_after_reopen/volumes/t/manifest.bin
cce2c440f10a27fa60c4ef29ee75a95aefba252261af220a1db589363622c711  name_dropped_after_reopen/volumes/t/vol_00065dba8bb76933.vol
8324c8de2ed867e321488f428dbb7c96b5ea3de19ce47fb1ff11f282e4b7ea75  name_dropped_after_reopen/wal/checkpoint.meta
81ab2c894d192c9f172cda679669c5c8a3ed9f98bbae548c5f57d294ab159d4c  name_dropped_after_reopen/wal/wal-20261013-153409-lsn-16.log
