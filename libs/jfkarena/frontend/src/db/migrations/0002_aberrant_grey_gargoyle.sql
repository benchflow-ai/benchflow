ALTER TABLE "models" DROP CONSTRAINT "models_model_id_key";--> statement-breakpoint
ALTER TABLE "models" ADD COLUMN "user_id" varchar;--> statement-breakpoint
ALTER TABLE "models" ADD CONSTRAINT "models_user_id_model_id_key" UNIQUE("user_id","model_id");