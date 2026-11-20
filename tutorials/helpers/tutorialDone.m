function tutorialDone(ctx)
%TUTORIALDONE  Print the run time and close the console diary.
fprintf('\n=== %s finished in %.1f s; log: tutorials/output/%s.log ===\n', ctx.name, toc(ctx.tStart), ctx.name);
diary off
end
