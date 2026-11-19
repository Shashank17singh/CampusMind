import { Link } from "react-router-dom";
import { Button } from "../components/ui";

export function NotFoundPage() {
  return (
    <div className="mx-auto flex max-w-md flex-col items-center justify-center py-20 text-center">
      <p className="font-display text-6xl font-bold tracking-tight text-ink">404</p>
      <h1 className="mt-4 text-2xl font-bold tracking-tight">Page Not Found</h1>
      <p className="mt-2 text-sm text-muted">
        The page or academic resource you requested does not exist or has moved.
      </p>
      <div className="mt-6 flex flex-wrap gap-3">
        <Link to="/chat">
          <Button variant="primary">Return to Chat</Button>
        </Link>
        <Link to="/planner">
          <Button variant="secondary">View Planner</Button>
        </Link>
      </div>
    </div>
  );
}
