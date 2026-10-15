-- One row per successful critique. Used by POST /critique to enforce the
-- daily limit. Run once in the Supabase SQL editor.

create table if not exists public.critique_log (
    id serial primary key,
    user_id uuid not null references auth.users (id) on delete cascade,
    created_at timestamptz not null default now()
);

-- Speeds up "how many critiques has this user had since midnight?"
create index if not exists critique_log_user_id_created_at_idx
    on public.critique_log (user_id, created_at);

alter table public.critique_log enable row level security;

grant all on public.critique_log to service_role;
grant all on sequence public.critique_log_id_seq to service_role;
